import logging
import pandas as pd
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

from src.data_utils import DataManager
from src.utils import load_config

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GTEModel:
    """
    Encapsulates the training and inference logic for the GTE model.
    """
    def __init__(self, config):
        """
        Initializes the GTEModel with the project configuration.

        Args:
            config (dict): The project configuration.
        """
        self.config = config
        self.model = None
        logging.info("GTEModel initialized.")

    def load_model(self, for_training=True):
        """
        Loads the GTE model.

        Args:
            for_training (bool): If True, loads the base model for training. 
                                 If False, loads the fine-tuned model for inference.
        """
        model_config = self.config['gte_model']
        if for_training:
            model_path = self.config['paths']['gte_base_model']
            logging.info(f"Loading base GTE model from: {model_path}")
        else:
            model_path = self.config['paths']['gte_finetuned_output']
            logging.info(f"Loading fine-tuned GTE model from: {model_path}")
        
        self.model = SentenceTransformer(model_path, trust_remote_code=True)
        self.model.max_seq_length = model_config['max_seq_length']
        logging.info("GTE model loaded successfully.")

    def train(self):
        """Fine-tunes the GTE model using Triplet Loss."""
        if not self.model:
            self.load_model(for_training=True)

        logging.info("--- Starting GTE Model Fine-tuning ---")

        # 1. Load Data
        data_manager = DataManager(self.config)
        triplet_dataset = data_manager.create_gte_triplet_dataset()
        train_dataloader = DataLoader(triplet_dataset, shuffle=True, batch_size=self.config['gte_model']['training']['batch_size'])
        logging.info(f"Loaded {len(triplet_dataset)} triplet samples for training.")

        # 2. Define Loss function
        loss = losses.TripletLoss(
            model=self.model,
            distance_metric='cosine', # Or 'euclidean'
            triplet_margin=self.config['gte_model']['training']['triplet_margin']
        )
        
        # 3. Configure and run training
        output_path = self.config['paths']['gte_finetuned_output']
        training_params = self.config['gte_model']['training']
        
        logging.info("Starting GTE training with model.fit()...")
        self.model.fit(
            train_objectives=[(train_dataloader, loss)],
            epochs=training_params['epochs'],
            warmup_steps=int(len(train_dataloader) * training_params['epochs'] * 0.1), # 10% warmup
            output_path=output_path,
            show_progress_bar=True,
            checkpoint_save_steps=len(train_dataloader), # Save checkpoint every epoch
            checkpoint_path=f"{output_path}/checkpoints/"
        )
        
        logging.info(f"GTE model training complete. Model saved to {output_path}")
        logging.info("--- GTE Model Fine-tuning Finished ---")

    def predict(self, texts_to_embed: list[str]) -> dict[str, list[float]]:
        """
        Generates embeddings for a list of texts.
        The centroid calculation and classification logic will be handled in the main inference pipeline.
        """
        if not self.model:
            self.load_model(for_training=False)

        logging.info(f"--- Generating GTE embeddings for {len(texts_to_embed)} texts ---")
        
        embeddings = self.model.encode(
            texts_to_embed,
            batch_size=self.config['gte_model']['inference']['batch_size'],
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        text_to_embedding = {text: emb for text, emb in zip(texts_to_embed, embeddings)}
        logging.info("--- GTE Embedding Generation Finished ---")
        return text_to_embedding

if __name__ == '__main__':
    """
    Main execution block to run the GTE training pipeline.
    """
    # 1. Load configuration
    config = load_config()
    
    # 2. Initialize the model handler
    gte_model = GTEModel(config)
    
    # 3. Run the training process
    gte_model.train()