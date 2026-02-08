import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import polars as pl
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from gnn_utils.dataset import TwhinDataset
from gnn_utils.models import TwhinModel
from gnn_utils.tracking import mlflow
from gnn_utils.utils import move_to_device, set_deterministic


def train_twhin_model(
    train_df,
    test_df,
    user_cardinality=3315,
    item_cardinality=25834,
    relation_cardinality=3,
    embedding_dim=128,
    batch_size=64,
    num_epochs=10,
    lr=0.001,
    reg_weight=0.01,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_log_dir="./logs",
    output_model_dir="./models",
    log_every_num_steps=10,
):
    logger = logging.getLogger(__name__)

    logger.info(f"Device used: {device}")

    logger.info(f"Number of users: {user_cardinality}")
    logger.info(f"Number of items: {item_cardinality}")
    logger.info(f"Number of action types: {relation_cardinality}")

    mlflow.set_experiment("TWHIN")
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(10)

    # Create datasets and loaders
    train_dataset = TwhinDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    test_dataset = TwhinDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

    # Initialize the model
    model = TwhinModel(
        embedding_dim=embedding_dim,
        item_cardinality=item_cardinality,
        user_cardinality=user_cardinality,
        relation_cardinality=relation_cardinality,
        reg_weight=reg_weight,
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup_epochs = 3
    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)

    # Create output directory if it doesn't exist
    os.makedirs(output_log_dir, exist_ok=True)
    os.makedirs(output_model_dir, exist_ok=True)

    batches_per_epoch = len(train_loader)

    prev_test_loss = None

    with mlflow.start_run(run_name=f"TwHIN:{datetime.now().strftime('%Y_%m_%d:%H_%M:%S')}"):
        mlflow.log_params(
            {
                "user_cardinality": user_cardinality,
                "item_cardinality": item_cardinality,
                "relation_cardinality": relation_cardinality,
                "embedding_dim": embedding_dim,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "lr": lr,
                "reg_weight": reg_weight,
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
                loss, loss_reg = model(batch)
                loss_total = loss + loss_reg
                loss_total.backward()
                optimizer.step()
                train_epoch_loss += loss_total.item()
                intermediate_losses.append(loss_total.item())
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
            test_epoch_mrr = 0.0
            test_batches = 0

            with torch.no_grad():
                for batch in tqdm(test_loader):
                    batch = move_to_device(batch, device)
                    loss, _, mrr = model(batch, calculate_mrr=True)

                    test_epoch_loss += loss.item()
                    test_epoch_mrr += mrr
                    test_batches += 1

            avg_test_loss = test_epoch_loss / test_batches
            avg_test_mrr = test_epoch_mrr / test_batches
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {avg_test_loss:.6f}, Test MRR: {avg_test_mrr:.6f}")
            mlflow.log_metrics(
                {"Test Loss": avg_test_loss, "Test MRR": avg_test_mrr}, step=(epoch + 1) * batches_per_epoch
            )
            if prev_test_loss is None or avg_test_loss < prev_test_loss:
                prev_test_loss = avg_test_loss
                with torch.no_grad():
                    torch.save(model.item_embeddings, os.path.join(output_model_dir, "item_embeddings.pt"))
                with open(os.path.join(output_log_dir, "twhin_final.json"), "w") as f:
                    json.dump({"Test Loss": avg_test_loss, "Test MRR": avg_test_mrr}, f, indent=2)
            else:
                logger.info("Test metric have not improved for 1 epoch, finishing run")
                logger.info(f"Saved metrics to {os.path.join(output_log_dir, 'twhin_final.json')}")
                logger.info(f"Saved embeddings to {os.path.join(output_model_dir, 'item_embeddings.pt')}")
                mlflow.log_artifact(
                    os.path.join(output_model_dir, "item_embeddings.pt"),
                    artifact_path=os.path.basename(os.path.normpath(output_model_dir)),
                )
                break


if __name__ == "__main__":
    set_deterministic()
    # Command line argument setup
    parser = argparse.ArgumentParser(description="Training TwhinModel")

    # Data
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)

    # Cardinalities
    parser.add_argument("--user-cardinality", type=int, required=True)
    parser.add_argument("--item-cardinality", type=int, required=True)
    parser.add_argument("--relation-cardinality", type=int, required=True)

    # Model parameters
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--reg-weight", type=float, default=0.01, help="Regularization weight")

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

    if os.path.exists(args.output_model_dir + "/item_embeddings.pt") and args.use_cached_results:
        logger.info("Item embeddings already exist, skipping training")
        exit(0)

    logger.info("Starting TwhinModel training")
    logger.info(f"Arguments: {args}")

    # Data loading
    logger.info(f"Loading training data from {args.train_data}")
    train_df = pl.read_parquet(args.train_data)

    logger.info(f"Loading test data from {args.test_data}")
    test_df = pl.read_parquet(args.test_data)

    # Start training
    train_twhin_model(
        train_df=train_df,
        test_df=test_df,
        user_cardinality=args.user_cardinality,
        item_cardinality=args.item_cardinality,
        relation_cardinality=args.relation_cardinality,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.learning_rate,
        reg_weight=args.reg_weight,
        device=args.device,
        output_log_dir=args.output_log_dir,
        output_model_dir=args.output_model_dir,
        log_every_num_steps=args.log_every_num_steps,
    )

    logger.info("Training completed successfully")
