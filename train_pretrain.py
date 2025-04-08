import warnings
warnings.filterwarnings("ignore")

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import polars as pl
import argparse
import logging
import os
from tqdm import tqdm
import numpy as np
import random

from utils.models import ModelBackbone, PretrainModel 


def set_deterministic(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

class PretrainDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.row(idx)
        product_id, action_type, candidate = row[0], row[2], row[3]
        
        return {
            "product_id": torch.tensor(product_id, dtype=torch.long),
            "action_type": torch.tensor(action_type, dtype=torch.long),
            "candidate": torch.tensor(candidate, dtype=torch.long)
        }


def collate_fn(batch):
    return {
        "product_id": pad_sequence([item["product_id"] for item in batch], batch_first=True, padding_value=0).to(torch.long),
        "action_type": pad_sequence([item["action_type"] for item in batch], batch_first=True, padding_value=0).to(torch.long),
        "candidate": torch.cat([item["candidate"] for item in batch])
    }


def train_pretrain_model(mode,
                         item_embeddings,
                         train_df, 
                         test_df, 
                         item_cardinality=25834,
                         embedding_dim=128, 
                         batch_size=64, 
                         num_epochs=10,
                         lr=0.001, 
                         device="cuda" if torch.cuda.is_available() else "cpu",
                         output_log_dir="./tmp/logs",
                         output_model_dir="./tmp/models"):
    logger = logging.getLogger(__name__)

    if os.path.exists(os.path.join(output_model_dir, f'backbone_after_pretrain_{mode}.pt')):
        logger.info("Already exist, skipping training")
        return 
    
    # Create datasets and loaders
    train_dataset = PretrainDataset(train_df)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    test_dataset = PretrainDataset(test_df)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    # Initialize the model
    backbone = ModelBackbone(
        embedding_dim=embedding_dim,
        item_cardinality=item_cardinality,
        item_embeddings=item_embeddings
    )
    model = PretrainModel(backbone).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup_epochs = 3
    scheduler_warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        total_iters=warmup_epochs
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_log_dir, exist_ok=True)
    os.makedirs(output_model_dir, exist_ok=True)
    
    prev_test_loss = None 
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_epoch_loss = 0.0
        train_batches = 0
        
        for batch in tqdm(train_loader):
            for key in batch:
                batch[key] = batch[key].to(device)
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
            train_batches += 1
        scheduler_warmup.step()
        
        avg_train_loss = train_epoch_loss / train_batches
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}")
        
        model.eval()
        test_epoch_loss = 0.0
        test_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader):
                for key in batch:
                    batch[key] = batch[key].to(device)
                loss = model(batch)
                
                test_epoch_loss += loss.item()
                test_batches += 1
        
        avg_test_loss = test_epoch_loss / test_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {avg_test_loss:.6f}")
        if prev_test_loss is None or avg_test_loss < prev_test_loss:
            prev_test_loss = avg_test_loss
            with torch.no_grad():
                torch.save(backbone, os.path.join(output_model_dir, f'backbone_after_pretrain_{mode}.pt'))
        else:
            break
    

if __name__ == "__main__":
    set_deterministic()
    # Command line argument setup
    parser = argparse.ArgumentParser(description="Training PretrainModel")
    
    # Data
    parser.add_argument('--train-data', type=str, required=True)
    parser.add_argument('--test-data', type=str, required=True)

    # Cardinalities
    parser.add_argument('--item-cardinality', type=int, required=True)

    # Model parameters
    parser.add_argument('--embedding-dim', type=int, default=128, help='Embedding dimension')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device for training (cuda/cpu)')
    
    # Output parameters
    parser.add_argument('--output-log-dir', type=str, default='./output', help='Directory for saving results')
    parser.add_argument('--output-model-dir', type=str, default='./output', help='Directory for saving models')
    
    args = parser.parse_args()
    
    # Logger setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting PretrainModel training")
    logger.info(f"Arguments: {args}")
    
    # Data loading
    logger.info(f"Loading training data from {args.train_data}")
    train_df = pl.read_parquet(args.train_data)
    
    
    logger.info(f"Loading test data from {args.test_data}")
    test_df = pl.read_parquet(args.test_data)
    
    # Start training
    for init in ['frozen', 'random', 'twhin']:
        if init == 'frozen':
            item_embeddings = torch.load(os.path.join(args.output_model_dir, 'item_embeddings.pt'))
            item_embeddings.weight.requires_grad = False
        elif init == 'random':
            item_embeddings = None 
        elif init == 'twhin':
            item_embeddings = torch.load(os.path.join(args.output_model_dir, 'item_embeddings.pt'))
            
        train_pretrain_model(
            mode=init,
            item_embeddings=item_embeddings,
            train_df=train_df,
            test_df=test_df,
            item_cardinality=args.item_cardinality,
            embedding_dim=args.embedding_dim,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.learning_rate,
            device=args.device,
            output_log_dir=args.output_log_dir,
            output_model_dir=args.output_model_dir
        )
    
        logger.info(f"Training {init} completed successfully")
