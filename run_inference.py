import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import load_config
from src.data_utils import DataManager, cleaner
from src.qwen_model import QwenModel
from src.gte_model import GTEModel

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def def run_qwen_inference(config: dict, test_df: pd.DataFrame) -> list:
    """
    Runs inference using the fine-tuned Qwen model.
    """
    logging.info("--- Starting Qwen Model Inference ---")
    qwen_model = QwenModel(config)
    
    # This is currently a placeholder. A full implementation requires loading the LoRA adapter.
    predictions = qwen_model.predict(test_df)
    logging.info("--- Qwen Model Inference Finished ---")
    return predictions


def run_gte_inference(config: dict, test_df: pd.DataFrame) -> list:
    """
    Runs inference using the fine-tuned GTE model and centroid-based classification.
    """
    logging.info("--- Starting GTE Model Inference ---")
    
    gte_model = GTEModel(config)
    data_manager = DataManager(config)

    # 1. Collect all unique texts to be embedded
    all_texts = data_manager.collect_unique_texts(test_df)
    
    # 2. Generate embeddings
    text_to_embedding = gte_model.predict(list(all_texts))

    # 3. Calculate centroids for each rule
    rule_centroids = {}
    for rule_text in tqdm(test_df['rule'].unique(), desc="Calculating Centroids"):
        rule_df = test_df[test_df['rule'] == rule_text]
        clean_rule = cleaner(str(rule_text))
        
        # In the dataset, 'positive_example' VIOLATES the rule, 'negative_example' DOES NOT.
        violating_texts = {cleaner(str(row[col])) for _, row in rule_df.iterrows() for col in ['positive_example_1', 'positive_example_2'] if pd.notna(row[col])}
        non_violating_texts = {cleaner(str(row[col])) for _, row in rule_df.iterrows() for col in ['negative_example_1', 'negative_example_2'] if pd.notna(row[col])}
        
        violating_vectors = [text_to_embedding[txt] for txt in violating_texts if txt in text_to_embedding]
        non_violating_vectors = [text_to_embedding[txt] for txt in non_violating_texts if txt in text_to_embedding]
        
        # The rule text itself is an anchor for the "non-violating" concept
        if clean_rule in text_to_embedding:
            non_violating_vectors.append(text_to_embedding[clean_rule])
        
        embedding_dim = gte_model.model.get_sentence_embedding_dimension()
        rule_centroids[clean_rule] = {
            'violating_centroid': np.mean(violating_vectors, axis=0) if violating_vectors else np.zeros(embedding_dim),
            'non_violating_centroid': np.mean(non_violating_vectors, axis=0) if non_violating_vectors else np.zeros(embedding_dim)
        }

    # 4. Classify test comments
    predictions = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Classifying with GTE"):
        clean_rule = cleaner(str(row['rule']))
        clean_body = cleaner(str(row['body']))
        
        if clean_body not in text_to_embedding:
            predictions.append(0) # Default to non-violating if comment not in map
            continue
            
        comment_vec = text_to_embedding[clean_body].reshape(1, -1)
        violating_centroid = rule_centroids[clean_rule]['violating_centroid'].reshape(1, -1)
        non_violating_centroid = rule_centroids[clean_rule]['non_violating_centroid'].reshape(1, -1)
        
        # Higher cosine similarity means closer
        dist_to_violating = cosine_similarity(comment_vec, violating_centroid)[0][0]
        dist_to_non_violating = cosine_similarity(comment_vec, non_violating_centroid)[0][0]
        
        is_violating = 1 if dist_to_violating > dist_to_non_violating else 0
        predictions.append(is_violating)

    logging.info("--- GTE Model Inference Finished ---")
    return predictions


def ensemble_predictions(qwen_preds: list, gte_preds: list, qwen_positive_token: str) -> list:
    """
    Ensembles predictions from Qwen and GTE models.
    The logic is: if Qwen predicts positive, use it, otherwise use GTE's prediction.
    """
    final_preds = []
    for q_pred, g_pred in zip(qwen_preds, gte_preds):
        if q_pred == qwen_positive_token:
            final_preds.append(1)
        else:
            final_preds.append(g_pred)
    return final_preds


def main():
    """
    Main function to run the full inference pipeline.
    """
    logging.info("====== Starting Full Inference Pipeline ======")
    
    # 1. Load config and data
    config = load_config()
    data_manager = DataManager(config)
    test_df = data_manager.load_test_data()
    
    # Optional: run on a smaller sample for debugging
    if config.get('debug_sample_size'):
        sample_size = config['debug_sample_size']
        logging.info(f"Using debug sample size: {sample_size}")
        test_df = test_df.head(sample_size).copy()

    # 2. Run GTE model inference
    gte_predictions = run_gte_inference(config, test_df)
    
    # 3. Run Qwen model inference (currently placeholder)
    qwen_predictions = run_qwen_inference(config, test_df)

    # 4. Ensemble the results
    qwen_positive_token = config['qwen_model']['prompt']['positive_token']
    final_predictions = ensemble_predictions(qwen_predictions, gte_predictions, qwen_positive_token)
    
    # 5. Save submission file
    submission_df = pd.DataFrame({'row_id': test_df['row_id'], 'rule_violation': final_predictions})
    output_path = config['paths']['submission_file']
    submission_df.to_csv(output_path, index=False)
    
    logging.info(f"Submission file saved to {output_path}")
    logging.info("====== Full Inference Pipeline Finished ======")


if __name__ == '__main__':
    main()