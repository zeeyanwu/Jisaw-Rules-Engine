import logging
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from src.utils import load_config

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_model(model_name: str, model_type: str):
    """
    Downloads a model and its tokenizer from Hugging Face and saves them to the local cache.

    Args:
        model_name (str): The name of the model on Hugging Face Hub.
        model_type (str): The type of the model ('qwen' or 'gte').
    """
    try:
        logging.info(f"--- Starting download for {model_type.upper()} model: {model_name} ---")

        if model_type == 'qwen':
            # For causal language models like Qwen, we download both model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            logging.info(f"Tokenizer for {model_name} downloaded successfully.")
            
            # We don't need to load the full model into VRAM just for downloading.
            # `from_pretrained` is smart enough to download and cache the files.
            # We can release the object immediately to free up memory.
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            logging.info(f"Model files for {model_name} downloaded successfully.")

        elif model_type == 'gte':
            # For sentence transformers, the library handles everything.
            model = SentenceTransformer(model_name)
            logging.info(f"SentenceTransformer model {model_name} downloaded successfully.")

        else:
            logging.error(f"Unknown model type: {model_type}")
            return

        # Clean up to be safe and release memory, although from_pretrained handles caching on disk.
        del model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()
        logging.info(f"--- Finished download for {model_name}. Resources released. ---")

    except Exception as e:
        logging.error(f"ERROR: Failed to download model {model_name}. Please check the model name and your network connection.")
        logging.error(f"Details: {e}")

if __name__ == "__main__":
    logging.info(">>> Starting model download process based on config.yaml <<<")
    
    # Load configuration from the single source of truth
    config = load_config()
    
    # Get model names from the config file
    qwen_model_name = config.get('qwen_model', {}).get('name')
    gte_model_name = config.get('gte_model', {}).get('name')
    
    if not qwen_model_name or not gte_model_name:
        logging.error("CRITICAL: Model names not found in config.yaml. Please check the file.")
    else:
        # Download Qwen model
        download_model(qwen_model_name, 'qwen')
        
        # Download GTE model
        download_model(gte_model_name, 'gte')

    logging.info(">>> Model download process finished. <<<")
