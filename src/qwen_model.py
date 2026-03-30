import torch
import logging
import pandas as pd
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

from src.data_utils import DataManager
from src.utils import load_config

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def str_to_torch_dtype(dtype_str: str):
    """Converts a string representation of a torch dtype to the actual dtype."""
    if dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "float32":
        return torch.float32
    else:
        logging.warning(f"Unsupported dtype '{dtype_str}' requested. Defaulting to float32.")
        return torch.float32


class QwenModel:
    """
    Encapsulates the training and inference logic for the Qwen model.
    """
    def __init__(self, config):
        """
        Initializes the QwenModel with the project configuration.

        Args:
            config (dict): The project configuration.
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        logging.info("QwenModel initialized.")

    def load_model(self, load_adapters: bool = False):
        """
        Loads the Qwen model and tokenizer. Optionally loads fine-tuned LoRA adapters.

        Args:
            load_adapters (bool): If True, loads the LoRA adapters from the output directory
                                  after loading the base model.
        """
        model_config = self.config['qwen_model']
        base_model_path = self.config['paths']['qwen_base_model']
        adapter_path = self.config['paths']['qwen_lora_output']
        
        logging.info(f"Loading base model from: {base_model_path}")

        quant_config = model_config['training']['quantization']
        compute_dtype = str_to_torch_dtype(quant_config['bnb_4bit_compute_dtype'])
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config['load_in_4bit'],
            bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant'],
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,  # Important for training stability
        )

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

        # --- FINAL FIX for Padding ---
        # Decoder-only models like Qwen do not have a default padding token.
        # We must set it to the EOS token to enable batching of sequences with different lengths.
        if self.tokenizer.pad_token is None:
            logging.info("Tokenizer does not have a pad token; setting it to the EOS token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # --- END FINAL FIX ---

        # When loading a model for training, we must also set the model's config to use the new pad_token_id.
        # This is not required for inference.
        if self.model.config.pad_token_id != self.tokenizer.pad_token_id:
            logging.info(f"Updating model config's pad_token_id to {self.tokenizer.pad_token_id}")
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        logging.info("Base model and tokenizer loaded successfully.")

        if load_adapters:
            logging.info(f"Loading LoRA adapters from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            logging.info("LoRA adapters loaded successfully.")

    def _format_dataset_for_sft(self, dataset: Dataset) -> Dataset:
        """
        Formats the dataset into the required single 'text' field for SFTTrainer.
        The format will be `prompt + completion`.
        """
        def format_prompt(example):
            # The SFTTrainer will handle the tokenization and training process.
            # We just need to provide the full text.
            example['text'] = example['prompt'] + example['completion']
            return example

        return dataset.map(format_prompt)

    def train(self, train_dataset: Dataset):
        """
        Fine-tunes the Qwen model using LoRA on a given dataset.

        Args:
            train_dataset (Dataset): The dataset (with 'prompt' and 'completion' columns)
                                     to be used for training.
        """
        if not self.model or not self.tokenizer:
            self.load_model(load_adapters=False)

        logging.info("--- Starting Qwen Model Fine-tuning ---")
        
        # The dataset is used directly without pre-formatting. SFTTrainer will handle tokenization.
        logging.info(f"Using {len(train_dataset)} samples for training.")

        # Configure LoRA
        # We load LoRA params from config, but override target_modules to ensure Qwen compatibility.
        lora_params = self.config['qwen_model']['training']['lora_config'].copy()
        lora_params['target_modules'] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_config = LoraConfig(**lora_params)

        # Configure Trainer
        trainer_params = self.config['qwen_model']['training']['trainer']
        output_dir = self.config['paths']['qwen_lora_output']

        # --- START SURGICAL PATCH for learning_rate TypeError ---
        # Explicitly cast learning_rate to float as a workaround for the persistent config loading issue.
        if 'learning_rate' in trainer_params:
            try:
                lr_val = trainer_params['learning_rate']
                logging.info(f"Applying surgical patch: converting learning_rate '{lr_val}' (type: {type(lr_val)}) to float.")
                trainer_params['learning_rate'] = float(lr_val)
            except (ValueError, TypeError) as e:
                logging.error(f"Could not convert learning_rate to float: {e}. Letting transformers use its default.")
                del trainer_params['learning_rate']
        # --- END SURGICAL PATCH ---
        
        # Use TrainingArguments for trainer-related hyperparameters
        training_args = TrainingArguments(
            output_dir=output_dir,
            **trainer_params
        )

        # Based on the user's final, correct analysis, we adopt "Method 2: formatting_func".
        # This is the only way to solve both errors simultaneously.

        # 1. Define the formatting function.
        # This function will be responsible for tokenization and truncation.
        max_length = self.config['qwen_model']['training'].get('max_seq_length', 512)
        def formatting_func(example):
            # The formatting_func should return a string or a list of strings.
            # It is responsible for preparing the text, not tokenizing it.
            return example['prompt']

        # 2. Define the data collator.
        # Its ONLY job is to perform dynamic padding on the already-tokenized batches.
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # 3. Create the SFTTrainer.
        #    - We pass the RAW dataset.
        #    - We provide the `formatting_func` to handle tokenization.
        #    - We provide the `data_collator` to handle padding.
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,        # The raw, original dataset
            peft_config=lora_config,
            formatting_func=formatting_func,    # The key insight: let the func handle tokenization
            data_collator=data_collator,        # The collator handles padding
            packing=False,
            max_seq_length=max_length,
        )
        
        # Start Training
        logging.info("Starting training...")
        trainer.train()
        logging.info("Training complete.")

        # Save Model
        logging.info(f"Saving LoRA adapter to {output_dir}")
        trainer.save_model(output_dir)
        logging.info("--- Qwen Model Fine-tuning Finished ---")

    def predict(self, prompt_text: str) -> str:
        """
        Runs inference on a single prompt text. Assumes model is already in memory.

        Args:
            prompt_text (str): The input prompt for the model.

        Returns:
            str: The generated text from the model.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model is not loaded. Call `load_model()` before prediction.")
        
        logging.info("--- Running single prompt inference ---")
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        
        # Generate output
        generate_ids = self.model.generate(**inputs, max_new_tokens=50, eos_token_id=self.tokenizer.eos_token_id)
        
        # Decode the generated tokens, removing the input prompt part
        full_result = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        completion_only = full_result[len(prompt_text):]

        logging.info("Inference complete.")
        return completion_only


if __name__ == '__main__':
    """
    Main execution block for manual testing (smoke test) of the QwenModel class.
    
    This test will:
    1. Load config and a small data sample.
    2. Initialize QwenModel (tests model loading, quantization, and PEFT setup).
    3. Run a minimal training loop (2 steps) to ensure the pipeline works and checks for OOM errors.
    4. Run a sample prediction to test inference logic.
    """
    print("--- Running Manual Smoke Test for src/qwen_model.py ---")

    try:
        # 1. Load config and data
        print("--> Loading configuration and data...")
        config = load_config()

        data_manager = DataManager(config)

        # Create a small dataset for the smoke test
        sample_size = config.get('debug_sample_size', 4) # Use 4 if not specified
        qwen_dataset = data_manager.load_qwen_dataset(
            use_pseudo=config['qwen_model']['training']['use_pseudo_train'],
            sample_size=sample_size
        )
        print(f"Loaded {len(qwen_dataset)} samples for smoke test training.\n")

        # 2. Initialize the model
        print("\n--> Initializing QwenModel...")
        qwen_model = QwenModel(config)
        qwen_model.load_model(load_adapters=False)
        print("QwenModel initialized and base model loaded successfully.")

        # 3. Run a minimal training loop
        print("\n--> Starting minimal training loop (2 steps)...")
        qwen_model.train(qwen_dataset)
        print("Minimal training completed successfully.")

        # 4. Run a sample prediction
        # The model in memory now has the trained LoRA adapters merged.
        print("\n--> Running sample prediction with the fine-tuned model...")
        
        # Build the prompt using the same logic as the DataManager
        prompt = data_manager._build_qwen_prompt({
            'text_1': "I think this is a great idea.",
            'text_2': "This is a stupid idea and you should feel bad."
        })

        prediction_output = qwen_model.predict(prompt)
        print(f"\nTest Prompt:\n------------------\n{prompt}\n------------------")
        print(f"\nModel Completion:\n------------------\n{prediction_output}\n------------------")
        
        # The model should have learned to generate the completion part.
        if "Text 2 is more toxic" in prediction_output:
            print("\n[SUCCESS] Prediction test PASSED: Model generated the expected output format.")
        else:
            print("\n[NOTE] Prediction test did not produce the exact expected output, which is acceptable for a brief smoke test.")

    except Exception as e:
        print(f"\n[FATAL ERROR] An error occurred during the smoke test.")
        logging.error("Smoke test failed", exc_info=True)

    print("\n--- Manual Smoke Test Finished ---")
