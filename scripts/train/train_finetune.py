import argparse
import json
import logging
import os
import warnings
from datetime import datetime

import numpy as np
import polars as pl
import torch
import torch.optim as optim
from sklearn.metrics import ndcg_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from gnn_utils.dataset import FinetuneDataset
from gnn_utils.dataset import finetune_collate_fn as collate_fn
from gnn_utils.models import FinetuneModel
from gnn_utils.tracking import mlflow
from gnn_utils.utils import move_to_device, set_deterministic


warnings.filterwarnings("ignore")


def evalutate_model(backbone, test_dataset, device):
    eval_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=2, collate_fn=collate_fn
    )
    backbone = backbone.to(device)
    backbone.eval()
    test_batches = 0
    test_epoch_ndcg = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            batch = move_to_device(batch, device)
            backbone_output = backbone(batch)
            user_embeddings = backbone_output["user_embeddings"]
            candidates_embeddings = backbone_output["candidates_embeddings"]
            candidates_lengths = batch["candidate_lengths"]

            scores = torch.einsum(
                "te,te->t", torch.repeat_interleave(user_embeddings, candidates_lengths, dim=0), candidates_embeddings
            )
            true_relevance = batch["candidate_mask"].cpu().numpy().astype(int)
            predicted_relevance = scores.cpu().numpy()
            test_epoch_ndcg += ndcg_score(true_relevance[None,], predicted_relevance[None,], k=10, ignore_ties=True)

            test_batches += 1

    return test_epoch_ndcg / test_batches


def train_finetune_model(
    mode,
    backbone,
    train_df,
    test_df,
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

    if os.path.exists(os.path.join(output_log_dir, f"finetune_{mode}_final.json")):
        with open(os.path.join(output_log_dir, f"finetune_{mode}_final.json"), "r") as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = []
    else:
        all_results = []
    if isinstance(all_results, dict):
        all_results = [all_results]

    mlflow.set_experiment("Finetune")
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(10)

    # Create datasets and loaders
    train_dataset = FinetuneDataset(train_df)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, collate_fn=collate_fn
    )

    test_dataset = FinetuneDataset(test_df)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=2, collate_fn=collate_fn
    )

    if os.path.exists(os.path.join(output_model_dir, f"backbone_after_finetune_{mode}.pt")) and use_cached_results:
        logger.info("Already exist, skipping training")
        test_ndcg = evalutate_model(
            torch.load(os.path.join(output_model_dir, f"backbone_after_finetune_{mode}.pt"), weights_only=False),
            test_dataset,
            device,
        )
        logger.info(f"Final Test NDCG: {test_ndcg:.6f}")
        return test_ndcg

    # Initialize the model
    model = FinetuneModel(backbone).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup_epochs = 3
    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)

    # Create output directory if it doesn't exist
    os.makedirs(output_log_dir, exist_ok=True)
    os.makedirs(output_model_dir, exist_ok=True)

    batches_per_epoch = len(train_loader)
    prev_test_loss = None

    with mlflow.start_run(run_name=f"{mode}:{datetime.now().strftime('%Y_%m_%d:%H_%M:%S')}"):
        mlflow.log_params(
            {
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
                    loss = model(batch)

                    test_epoch_loss += loss.item()
                    test_batches += 1

            avg_test_loss = test_epoch_loss / test_batches

            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {avg_test_loss:.6f}")
            mlflow.log_metrics({"Test Loss": avg_test_loss}, step=(epoch + 1) * batches_per_epoch)
            if prev_test_loss is None or avg_test_loss < prev_test_loss:
                prev_test_loss = avg_test_loss
                with torch.no_grad():
                    logger.info(
                        f"Saving model checkpoint to {os.path.join(output_model_dir, f'backbone_after_finetune_{mode}.pt')}"
                    )
                    torch.save(backbone, os.path.join(output_model_dir, f"backbone_after_finetune_{mode}.pt"))
            else:
                break
        mlflow.log_artifact(
            os.path.join(output_model_dir, f"backbone_after_finetune_{mode}.pt"),
            artifact_path=os.path.basename(os.path.normpath(output_model_dir)),
        )

    test_ndcg = evalutate_model(
        torch.load(os.path.join(output_model_dir, f"backbone_after_finetune_{mode}.pt"), weights_only=False),
        test_dataset,
        device,
    )
    with open(os.path.join(output_log_dir, f"finetune_{mode}_final.json"), "w") as f:
        json.dump(
            all_results + [{"Test Loss": prev_test_loss, "nDCG@10": test_ndcg}],
            f,
            indent=2,
        )
    logger.info(f"Final Test nDCG: {test_ndcg:.6f}")
    logger.info(f"Saved metrics to {os.path.join(output_log_dir, f'finetune_{mode}_final.json')}")
    return test_ndcg


if __name__ == "__main__":
    set_deterministic()
    # Command line argument setup
    parser = argparse.ArgumentParser(description="Training FinetuneModel")

    # Data
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)

    # Training parameters
    parser.add_argument(
        "--use-cached-results",
        action="store_true",
        help="Training will not be launched again if the checkpoint exists",
    )

    # Training parameters
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

    logger.info("Starting FinetuneModel training")

    # Data loading
    logger.info(f"Loading training data from {args.train_data}")
    train_df = pl.read_parquet(args.train_data)

    logger.info(f"Loading test data from {args.test_data}")
    test_df = pl.read_parquet(args.test_data)

    for init in ["frozen", "random", "twhin"]:
        backbone = torch.load(
            os.path.join(args.output_model_dir, f"backbone_after_pretrain_{init}.pt"), weights_only=False
        )
        if init == "frozen":
            backbone.item_embeddings.weight.requires_grad = False

        train_finetune_model(
            mode=init,
            backbone=backbone,
            train_df=train_df,
            test_df=test_df,
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
