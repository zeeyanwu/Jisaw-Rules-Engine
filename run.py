
import argparse
import pandas as pd
import os

# It's good practice to wrap the core logic in a src package.
# This makes imports cleaner and the project structure more standard.
from src import qwen_model, gte_model, config

def main():
    parser = argparse.ArgumentParser(description="Run pipeline for Jigsaw Community Rules Challenge.")
    parser.add_argument(
        "--steps", 
        nargs='+', 
        choices=["train_qwen", "infer_qwen", "train_gte", "infer_gte", "ensemble"],
        help="Specify which steps of the pipeline to run.",
        required=True
    )
    parser.add_argument(
        "--qwen_gpus", 
        type=int, 
        default=1, 
        help="Number of GPUs to use for Qwen inference (uses multiprocessing)."
    )
    parser.add_argument(
        "--gte_metric", 
        type=str, 
        default="cosine", 
        choices=["cosine", "euclidean"],
        help="Distance metric for GTE centroid classification."
    )
    parser.add_argument(
        "--ensemble_weights", 
        nargs=2, 
        type=float, 
        default=[0.6, 0.4], 
        help="Weights for Qwen and GTE submissions in the format: --ensemble_weights <qwen_weight> <gte_weight>"
    )

    args = parser.parse_args()

    print(f"Running selected steps: {args.steps}")

    # --- Create necessary directories ---
    # Ensure model and submission directories exist before running any steps
    os.makedirs(os.path.dirname(config.QWEN_LORA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(config.GTE_FINETUNED_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(config.FINAL_SUBMISSION_PATH), exist_ok=True)
    
    if "train_qwen" in args.steps:
        print("\n===== [Step: Training Qwen Model] =====")
        qwen_model.train()
    
    if "infer_qwen" in args.steps:
        print("\n===== [Step: Qwen Model Inference] =====")
        qwen_model.infer(num_gpus=args.qwen_gpus)
        
    if "train_gte" in args.steps:
        print("\n===== [Step: Training GTE Model] =====")
        gte_model.train()
        
    if "infer_gte" in args.steps:
        print("\n===== [Step: GTE Model Inference] =====")
        use_cosine = (args.gte_metric == "cosine")
        gte_model.infer(use_cosine_similarity=use_cosine)
        
    if "ensemble" in args.steps:
        print("\n===== [Step: Ensembling Results] =====")
        print(f"Reading Qwen submission from: {config.QWEN_SUBMISSION_PATH}")
        print(f"Reading GTE submission from: {config.GTE_SUBMISSION_PATH}")
        
        try:
            sub_llm = pd.read_csv(config.QWEN_SUBMISSION_PATH)
            sub_gte = pd.read_csv(config.GTE_SUBMISSION_PATH)
        except FileNotFoundError as e:
            print(f"Error: {e}. One of the submission files is missing. Please run inference for both models before ensembling.")
            return

        # Merge and weight the predictions
        merged_df = pd.merge(sub_llm, sub_gte, on="row_id", suffixes=["_llm", "_gte"])
        qwen_weight, gte_weight = args.ensemble_weights
        print(f"Ensembling with weights: Qwen={qwen_weight}, GTE={gte_weight}")
        
        merged_df["rule_violation"] = (merged_df["rule_violation_llm"] * qwen_weight + 
                                       merged_df["rule_violation_gte"] * gte_weight)
        
        # Binarize the result based on a 0.5 threshold
        merged_df["rule_violation"] = (merged_df["rule_violation"] > 0.5).astype(int)
        
        final_submission = merged_df[["row_id", "rule_violation"]]
        
        final_submission.to_csv(config.FINAL_SUBMISSION_PATH, index=False)
        print(f"Ensemble complete. Final submission saved to: {config.FINAL_SUBMISSION_PATH}")

    print("\nAll specified steps completed.")

if __name__ == "__main__":
    main()
