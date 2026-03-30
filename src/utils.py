import yaml
import re
from pathlib import Path
from functools import reduce
import operator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _get_from_dict(data_dict, keys):
    """Access a nested dictionary using a list of keys."""
    try:
        return reduce(operator.getitem, keys, data_dict)
    except (KeyError, TypeError):
        return None

def _resolve_config_vars(config: dict) -> dict:
    """A robust, iterative config variable resolver that preserves types."""
    placeholder_pattern = re.compile(r'\$\{(.*?)\}')

    def substitute(node, original_config):
        """Recursively traverses a node and substitutes placeholders using the original config for lookups."""
        if isinstance(node, dict):
            return {k: substitute(v, original_config) for k, v in node.items()}
        if isinstance(node, list):
            return [substitute(item, original_config) for item in node]
        if not isinstance(node, str):
            return node # It is not a string, so it cannot have placeholders.

        # --- This is the core logic for string substitution ---

        # Case 1: The entire string is a placeholder, e.g., "${qwen_model.training.trainer.learning_rate}"
        # In this case, we want to replace it with the actual value, preserving its type (e.g., float).
        full_match = placeholder_pattern.fullmatch(node.strip())
        if full_match:
            keys = full_match.group(1).split('.')
            replacement = _get_from_dict(original_config, keys)
            # If the replacement is another placeholder, we recursively resolve it.
            return substitute(replacement, original_config)

        # Case 2: The string CONTAINS a placeholder, e.g., "${paths.base_data}/train.csv"
        # In this case, the final output must be a string.
        def repl(match):
            keys = match.group(1).split('.')
            replacement = _get_from_dict(original_config, keys)
            # Recursively resolve the replacement in case it's another placeholder.
            final_value = substitute(replacement, original_config)
            return str(final_value) # Cast to string for concatenation.

        return placeholder_pattern.sub(repl, node)

    # Iteratively apply substitutions until the config is stable.
    # This handles nested dependencies (e.g., a -> b -> c).
    max_passes = 10
    resolved_config = config
    for _ in range(max_passes):
        new_config = substitute(resolved_config, config) # <<< KEY: Always use original `config` for lookups.
        if new_config == resolved_config:
            return new_config # Stable, no changes in this pass.
        resolved_config = new_config

    logging.warning(f"Config resolution did not stabilize after {max_passes} passes. Check for circular references.")
    return resolved_config

def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """
    Loads a YAML configuration file and resolves internal variable placeholders.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: The loaded and resolved configuration.
    """
    config_path_obj = Path(config_path)
    if not config_path_obj.is_file():
        logging.error(f"Configuration file not found at {config_path_obj}")
        raise FileNotFoundError(f"Configuration file not found at: {config_path_obj}")

    try:
        with open(config_path_obj, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise

    return _resolve_config_vars(config)

if __name__ == '__main__':
    """
    Main execution block for manual testing.
    This allows the script to be run directly to test its functionality.
    e.g., python -m src.utils
    """
    print("--- Running Manual Test for src/utils.py ---")
    try:
        # Attempt to load the default configuration file and resolve variables
        config = load_config()
        print("Configuration loaded and variables resolved successfully!")
        
        # Print a few sample keys to verify content and resolution
        print("\nSample resolved configuration values:")
        if 'paths' in config:
            print(f"  Train CSV path: {config['paths'].get('train_csv')}")
            print(f"  Test CSV path: {config['paths'].get('test_csv')}")
            print(f"  Qwen base model path: {config['paths'].get('qwen_base_model')}")
            print(f"  Qwen LoRA output path: {config['paths'].get('qwen_lora_output')}")

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("Please ensure you are running this script from the project root directory, ")
        print("and the 'configs/config.yaml' file exists.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    print("\n--- Manual Test Finished ---")
