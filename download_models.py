
import logging
from src.utils import load_config
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import os

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def download_qwen_model(model_name: str, config: dict):
    """
    Downloads the Qwen model and tokenizer from Hugging Face Hub.
    Includes quantization configuration to ensure all necessary components are checked.
    """
    logging.info(f"Starting download for Qwen model: {model_name}")
    try:
        # We don't need to load the full model into memory, 
        # just instantiating it with from_pretrained will trigger the download.
        # This also ensures that if files are already cached, they won't be re-downloaded.
        
        # To avoid using too much RAM, we can download configuration first,
        # then the model files. But from_pretrained handles this efficiently.
        # We specify device_map to avoid loading the model onto GPU during download.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="cpu" # Ensure model isn't loaded onto GPU
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Clean up to free memory
        del model
        del tokenizer

        logging.info(f"Successfully downloaded and cached Qwen model: {model_name}")
        # The cache path is typically ~/.cache/huggingface/hub/models--<model_name_with--slashes>
        # We can find the exact path if needed, but for now, just confirming download is enough.
        
    except Exception as e:
        logging.error(f"Failed to download Qwen model '{model_name}'. Error: {e}")
        logging.error("Please check the model name in 'configs/config.yaml', your internet connection, and Hugging Face Hub status.")

def download_gte_model(model_name: str):
    """
    Downloads the GTE model from Hugging Face Hub.
    """
    logging.info(f"Starting download for GTE model: {model_name}")
    try:
        # Similarly, instantiating SentenceTransformer will trigger the download.
        model = SentenceTransformer(model_name, device='cpu')
        
        # Clean up
        del model

        logging.info(f"Successfully downloaded and cached GTE model: {model_name}")

    except Exception as e:
        logging.error(f"Failed to download GTE model '{model_name}'. Error: {e}")
        logging.error("Please check the model name in 'configs/config.yaml', your internet connection, and Hugging Face Hub status.")

if __name__ == '__main__':
    """
    Main execution block to download all required models for the project.
    """
    logging.info("--- Starting Model Download Process ---")
    
    try:
        config = load_config()
        
        # --- Download Qwen Model ---
        qwen_model_name = config.get('paths', {}).get('qwen_base_model')
        if qwen_model_name:
            download_qwen_model(qwen_model_name, config)
        else:
            logging.warning("Qwen model name not found in 'configs/config.yaml' under paths.qwen_base_model. Skipping.")
            
        # --- Download GTE Model ---
        gte_model_name = config.get('paths', {}).get('gte_base_model')
        if gte_model_name:
            download_gte_model(gte_model_name)
        else:
            logging.warning("GTE model name not found in 'configs/config.yaml' under paths.gte_base_model. Skipping.")

    except FileNotFoundError:
        logging.fatal("Configuration file 'configs/config.yaml' not found. Cannot proceed.")
        logging.fatal("Please ensure you are running this script from the project root directory.")
    except Exception as e:
        logging.fatal(f"An unexpected error occurred: {e}")

    logging.info("--- Model Download Process Finished ---")

