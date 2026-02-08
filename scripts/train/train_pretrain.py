import argparse
import json
import logging
import os
import warnings
from datetime import datetime

import mlflow
import numpy as np
import polars as pl
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from gnn_utils.dataset import PretrainDataset
from gnn_utils.dataset import pretrain_collate_fn as collate_fn
from gnn_utils.models import ModelBackbone, PretrainModel
from gnn_utils.utils import move_to_device, set_deterministic


warnings.filterwarnings("ignore")


def train_pretrain_model(
    mode,
    item_embeddings,
    train_df,
    test_df,
    item_cardinality=25834,
    embedding_dim=128,
    batch_size=64,
    num_epochs=10,
    lr=0.001,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_log_dir="./logs",
    output_model_dir="./models",
    log_every_num_steps=10,
    use_cached_results=False,
):
    logger = logging.getLogger(__name__)

    mlflow.set_experiment("Pretrain")
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(10)

    if os.path.exists(os.path.join(output_model_dir, f"backbone_after_pretrain_{mode}.pt")) and use_cached_results:
        logger.info(
            f"{os.path.join(output_model_dir, f'backbone_after_pretrain_{mode}.pt')} already exists, skipping training"
        )
        return

    # Create datasets and loaders
    train_dataset = PretrainDataset(train_df)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, collate_fn=collate_fn
    )

    test_dataset = PretrainDataset(test_df)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=2, collate_fn=collate_fn
    )

    # Initialize the model
    backbone = ModelBackbone(
        embedding_dim=embedding_dim, item_cardinality=item_cardinality, item_embeddings=item_embeddings
    )
    model = PretrainModel(backbone).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup_epochs = 3
    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)

    # Create output directory if it doesn't exist
    os.makedirs(output_log_dir, exist_ok=True)
    os.makedirs(output_model_dir, exist_ok=True)

    batches_per_epoch = len(train_loader)

    best_loss = None

    with mlflow.start_run(run_name=f"{mode}:{datetime.now().strftime('%Y_%m_%d:%H_%M:%S')}"):
        mlflow.log_params(
            {
                "item_cardinality": item_cardinality,
                "embedding_dim": embedding_dim,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "lr": lr,
            }
        )

        for epoch in range(num_epochs):
            # Training
            model.train()
            train_epoch_loss = 0.0
            train_batches = 0
            intermediate_losses = []

            pbar = tqdm(train_loader)

            for batch in pbar:
                batch = move_to_device(batch, device)
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()
                intermediate_losses.append(loss.item())
                if (train_batches % log_every_num_steps == 0) or (train_batches == batches_per_epoch - 1):
                    pbar.set_description(f"Epoch {epoch + 1}, Train Loss: {train_epoch_loss / (train_batches + 1):.4f}")
                    mlflow.log_metrics(
                        {"Train Loss": np.mean(intermediate_losses)},
                        step=epoch * batches_per_epoch + train_batches + 1,
                    )
                    intermediate_losses = []
                train_batches += 1
            scheduler_warmup.step()

            model.eval()
            test_epoch_loss = 0.0
            test_batches = 0

            with torch.no_grad():
                for batch in tqdm(test_loader):
                    batch = move_to_device(batch, device)
                    loss, recalls = model(batch, recall_at=(10, 50, 100))

                    test_epoch_loss += loss.item()
                    test_batches += 1

            recalls = {f"Recall:{key}": value for key, value in recalls.items()}
            avg_test_loss = test_epoch_loss / test_batches
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {avg_test_loss:.6f}, {', '.join([f'{key}: {value:.6f}' for key, value in recalls.items()])}"
            )
            mlflow.log_metrics(
                {"Test Loss": avg_test_loss} | recalls,
                step=(epoch + 1) * batches_per_epoch,
            )
            if best_loss is None or avg_test_loss < best_loss:
                best_loss = avg_test_loss
                with torch.no_grad():
                    torch.save(backbone, os.path.join(output_model_dir, f"backbone_after_pretrain_{mode}.pt"))
                with open(os.path.join(output_log_dir, f"pretrain_{mode}_final.json"), "w") as f:
                    json.dump({"Test Loss": avg_test_loss} | recalls, f, indent=2)
            else:
                break
        logger.info(f"Saved metrics to {os.path.join(output_log_dir, f'pretrain_{mode}_final.json')}")
        mlflow.log_artifact(
            os.path.join(output_model_dir, f"backbone_after_pretrain_{mode}.pt"),
            artifact_path=os.path.basename(os.path.normpath(output_model_dir)),
        )


if __name__ == "__main__":
    set_deterministic()
    # Command line argument setup
    parser = argparse.ArgumentParser(description="Training PretrainModel")

    # Data
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)

    # Cardinalities
    parser.add_argument("--item-cardinality", type=int, required=True)

    # Model parameters
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")

    # Training parameters
    parser.add_argument(
        "--use-cached-results",
        action="store_true",
        help="Training will not be launched again if the checkpoint exists",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training (cuda/cpu)",
    )

    # Output parameters
    parser.add_argument("--output-log-dir", type=str, default="./output", help="Directory for saving results")
    parser.add_argument("--output-model-dir", type=str, default="./output", help="Directory for saving models")

    # Logging
    parser.add_argument("--log-every-num-steps", type=int, default=10, help="The frequency to update pbar")

    args = parser.parse_args()

    # Logger setup
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Starting PretrainModel training")

    # Data loading
    logger.info(f"Loading training data from {args.train_data}")
    train_df = pl.read_parquet(args.train_data)

    logger.info(f"Loading test data from {args.test_data}")
    test_df = pl.read_parquet(args.test_data)

    # Start training
    for init in ["frozen", "random", "twhin"]:
        if init == "frozen":
            item_embeddings = torch.load(os.path.join(args.output_model_dir, "item_embeddings.pt"), weights_only=False)
            item_embeddings.weight.requires_grad = False
        elif init == "random":
            item_embeddings = None
        elif init == "twhin":
            item_embeddings = torch.load(os.path.join(args.output_model_dir, "item_embeddings.pt"), weights_only=False)

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
            output_model_dir=args.output_model_dir,
            log_every_num_steps=args.log_every_num_steps,
            use_cached_results=args.use_cached_results,
        )

        logger.info(f"Training {init} completed successfully")
