
import pandas as pd
import numpy as np
import random
import re
from urllib.parse import urlparse
from datasets import Dataset
from tqdm.auto import tqdm
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cleaner(text):
    """Replace URLs with a structured format for better processing by models."""
    if not isinstance(text, str):
        return text

    url_pattern = r'https?://[^\s<>\"{}|\\\\^`\\[\\]]+'

    def replace_url(match):
        url = match.group(0)
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().replace('www.', '')
            path_parts = [part for part in parsed.path.split('/') if part]
            if path_parts:
                # Keep the domain and the first one or two path components
                important_path = '/'.join(path_parts[:2])
                return f"<url>: ({domain}/{important_path})"
            else:
                return f"<url>: ({domain})"
        except Exception:
            # If parsing fails for any reason, return a generic placeholder
            return "<url>: (unknown)"

    return re.sub(url_pattern, replace_url, text)


class DataManager:
    """
    Handles all data loading, preprocessing, and dataset creation.
    """
    def __init__(self, config):
        """
        Initializes the DataManager with the project configuration.

        Args:
            config (dict): The project configuration loaded from config.yaml.
        """
        self.config = config
        np.random.seed(self.config['general']['seed'])
        random.seed(self.config['general']['seed'])
        logging.info("DataManager initialized.")

    def _build_qwen_prompt(self, row):
        """Builds the few-shot prompt for the Qwen model for a single row."""
        prompt_config = self.config['qwen_model']['prompt']

        # The text block is intentionally left-aligned to avoid leading spaces in the final prompt.
        prompt = f"""{prompt_config['system_prompt']}
Subreddit: r/{row["subreddit"]}
Rule: {row["rule"]}
Examples:
1) {row["positive_example"]}
{prompt_config['judge_words']} Yes
2) {row["negative_example"]}
{prompt_config['judge_words']} No
Comment: {row["body"]}
{prompt_config['judge_words']}"""
        return prompt

    def _get_pseudo_train_df(self):
        """
        Creates a pseudo-training dataframe by sampling from the test set and its examples.
        This is a form of semi-supervised learning to augment the training data.
        """
        logging.info("Creating pseudo training data from test set...")
        try:
            test_df = pd.read_csv(self.config['paths']['test_csv'])
        except FileNotFoundError:
            logging.error(f"Test data file not found at {self.config['paths']['test_csv']}")
            raise

        qwen_config = self.config['qwen_model']['training']
        
        # Sample a fraction of the test data
        sampled_df = test_df.groupby('rule', group_keys=False).apply(
            lambda x: x.sample(frac=qwen_config['pseudo_frac'], random_state=self.config['general']['seed'])
        ).reset_index(drop=True)
        logging.info(f"Selected {len(sampled_df)} samples from test data for pseudo-training.")

        merge_list = []
        for violation_type in ["positive", "negative"]:
            for i in range(1, 3):
                sub_df = sampled_df[[
                    "rule", "subreddit", "positive_example_1", "positive_example_2",
                    "negative_example_1", "negative_example_2"
                ]].copy()

                sub_df["body"] = sub_df[f"{violation_type}_example_{i}"]
                sub_df[f"{violation_type}_example"] = sub_df[f"{violation_type}_example_{3-i}"]

                anti_violation_type = "negative" if violation_type == "positive" else "positive"
                sub_df[f"{anti_violation_type}_example"] = np.where(
                    np.random.rand(len(sub_df)) < 0.5,
                    sub_df[f"{anti_violation_type}_example_1"],
                    sub_df[f"{anti_violation_type}_example_2"]
                )

                sub_df["rule_violation"] = 1 if violation_type == "positive" else 0
                sub_df.drop(columns=["positive_example_1", "positive_example_2", "negative_example_1", "negative_example_2"], inplace=True)
                merge_list.append(sub_df)

        return pd.concat(merge_list, axis=0).drop_duplicates(ignore_index=True)

    def load_qwen_dataset(self, use_pseudo=False, sample_size=None):
        """
        Builds a Hugging Face Dataset for the Qwen model.

        Args:
            use_pseudo (bool): Whether to use the pseudo-training data from the test set.
            sample_size (int, optional): If specified, subsample the dataset to this size for quick testing.

        Returns:
            Dataset: The prepared Hugging Face dataset.
        """
        logging.info("Loading data for Qwen model...")
        try:
            train_df = pd.read_csv(self.config['paths']['train_csv'])
        except FileNotFoundError:
            logging.error(f"Train data file not found at {self.config['paths']['train_csv']}")
            raise

        if use_pseudo:
            pseudo_df = self._get_pseudo_train_df()
            df = pd.concat([train_df, pseudo_df], ignore_index=True)
            logging.info(f"Combined train data ({len(train_df)}) and pseudo data ({len(pseudo_df)}). Total: {len(df)}")
        else:
            df = train_df

        # Randomly select one positive and one negative example
        df["positive_example"] = np.where(np.random.rand(len(df)) < 0.5, df["positive_example_1"], df["positive_example_2"])
        df["negative_example"] = np.where(np.random.rand(len(df)) < 0.5, df["negative_example_1"], df["negative_example_2"])
        
        df["prompt"] = df.apply(self._build_qwen_prompt, axis=1)
        
        prompt_config = self.config['qwen_model']['prompt']
        df["completion"] = df["rule_violation"].map({
            1: prompt_config['positive_token'],
            0: prompt_config['negative_token'],
        })
        
        if sample_size:
            logging.info(f"Subsampling dataset to {sample_size} examples.")
            df = df.head(sample_size)
            
        return Dataset.from_pandas(df[["prompt", "completion"]])

    def create_gte_triplet_dataset(self, test_df: pd.DataFrame | None = None) -> list[dict]:
        """
        Creates a dataset for triplet loss training with GTE.
        For each anchor, it finds a positive (same rule, different subreddit)
        and a negative (different rule) example.

        Args:
            test_df (pd.DataFrame | None, optional): A pre-loaded and sampled dataframe
                of test data. If None, the data will be loaded from the path specified
                in the config. Defaults to None.

        Returns:
            list[dict]: A list of triplet dictionaries.
        """
        logging.info("Creating GTE triplet dataset...")
        
        # If no dataframe is provided, load it from disk.
        if test_df is None:
            logging.info("No test_df provided, loading from disk...")
            test_df = self.load_test_data()
            # Subsample test_df for faster processing during dev/debug
            if self.sample_size is not None and self.sample_size < len(test_df):
                test_df = test_df.sample(n=self.sample_size, random_state=self.config.get('random_seed', 42))

        # The training data is used to find positive/negative pairs.
        train_df = pd.read_csv(self.config['paths']['train_csv'])
        
        triplets = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating GTE Triplets"):
            anchor_text = row['text']
            
            # Find a positive example: same rule, different subreddit
            positive_pool = train_df[(train_df['rule_id'] == row['rule_id']) & (train_df['subreddit'] != row['subreddit'])]
            if not positive_pool.empty:
                positive_text = positive_pool.sample(1)['text'].iloc[0]
            else:
                continue # Skip if no positive example found

            # Find a negative example: different rule
            negative_pool = train_df[train_df['rule_id'] != row['rule_id']]
            if not negative_pool.empty:
                negative_text = negative_pool.sample(1)['text'].iloc[0]
            else:
                continue # Skip if no negative example found
                
            triplets.append({'anchor': anchor_text, 'positive': positive_text, 'negative': negative_text})
            
        logging.info(f"Created {len(triplets)} triplets.")
        return triplets

    def load_test_data(self):
        """Loads and cleans the test data."""
        logging.info("Loading test data...")
        df = pd.read_csv(self.config['paths']['test_csv'])
        df['text'] = df['text'].apply(cleaner)
        logging.info(f"Loaded {len(df)} test samples.")
        return df

    def collect_unique_texts(self, df: pd.DataFrame) -> set:
        """Collects a set of unique texts from the 'text' column of a DataFrame."""
        logging.info("Collecting unique texts for embedding...")
        all_texts = set(df['text'].unique())
        logging.info(f"Found {len(all_texts)} unique texts to embed.")
        return all_texts


# ==============================================================================
# Professional Manual Testing Block
# ==============================================================================

def test_qwen_pipeline(data_manager, sample_size):
    """Tests the Qwen data loading and prompt generation."""
    print("\n--- Testing Qwen Data Pipeline ---")
    qwen_dataset = data_manager.load_qwen_dataset(use_pseudo=False, sample_size=sample_size)
    print(f"Successfully created Qwen dataset with {len(qwen_dataset)} samples.")
    
    if len(qwen_dataset) > 0:
        print("\nExample Qwen prompt:")
        print("----------------------")
        first_item = qwen_dataset[0]
        print(first_item['prompt'])
        print("----------------------")
        print(f"Expected completion: {first_item['completion']}")

def test_gte_pipeline(data_manager, sample_size):
    """Tests the GTE triplet generation without monkey-patching."""
    print("\n--- Testing GTE Triplet Data Pipeline ---")
    try:
        # Pre-load a small, clean sample of test data to pass to the method.
        # This makes the test clean, fast, and independent.
        test_df_sample = data_manager.load_test_data()
        if sample_size < len(test_df_sample):
             test_df_sample = test_df_sample.sample(n=sample_size, random_state=data_manager.config.get('random_seed', 42))

        # Pass the pre-loaded dataframe directly to the method.
        gte_dataset = data_manager.create_gte_triplet_dataset(test_df=test_df_sample)

        print(f"Successfully created GTE triplet dataset with {len(gte_dataset)} triplets.")
        if len(gte_dataset) > 0:
            print("\nExample GTE Triplet:")
            print("----------------------")
            first_triplet = gte_dataset[0]
            print(f"  Anchor: {first_triplet['anchor'][:80]}...")
            print(f"Positive: {first_triplet['positive'][:80]}...")
            print(f"Negative: {first_triplet['negative'][:80]}...")
            print("----------------------")

    except FileNotFoundError:
        print(f"[ERROR] Data file not found. Skipping GTE pipeline test.")
        print(f"  Expected path: {data_manager.config['paths']['test_csv']}")


if __name__ == '__main__':
    """
    Main execution block for professional, command-line driven manual testing.
    
    Usage:
        python -m src.data_utils                (Run all tests)
        python -m src.data_utils --test qwen    (Run only Qwen pipeline test)
        python -m src.data_utils --test gte     (Run only GTE pipeline test)
    """
    import argparse
    from src.utils import load_config

    parser = argparse.ArgumentParser(description="Manual tests for DataManager.")
    parser.add_argument(
        '--test', 
        type=str, 
        choices=['qwen', 'gte', 'all'], 
        default='all',
        help="Specify which pipeline to test: 'qwen', 'gte', or 'all'."
    )
    args = parser.parse_args()

    print("--- Running Manual Test for src/data_utils.py ---")

    try:
        config = load_config()
        data_manager = DataManager(config)
        sample_size = config.get('debug_sample_size', 10)
        print(f"Using debug sample size: {sample_size}")

        if args.test in ['qwen', 'all']:
            test_qwen_pipeline(data_manager, sample_size)
        
        if args.test in ['gte', 'all']:
            test_gte_pipeline(data_manager, sample_size)

    except FileNotFoundError as e:
        print(f"[FATAL ERROR] {e}")
        print("Please ensure you are running this from the project root and 'configs/config.yaml' exists.")
    except Exception as e:
        print(f"An unexpected fatal error occurred: {e}")

    print("\n--- Manual Test Finished ---")
