import argparse
import json
import logging
import os
import warnings

import polars as pl
import torch
import torch.optim as optim
from sklearn.metrics import ndcg_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import FinetuneDataset
from src.dataset import finetune_collate_fn as collate_fn
from src.models import FinetuneModel
from src.utils import move_to_device, set_deterministic


warnings.filterwarnings("ignore")


def evalutate_model(backbone, test_dataset, device):
    eval_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, collate_fn=collate_fn
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
    output_log_dir="./tmp/logs",
    output_model_dir="./tmp/models",
):
    logger = logging.getLogger(__name__)

    # Create datasets and loaders
    train_dataset = FinetuneDataset(train_df)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, collate_fn=collate_fn
    )

    test_dataset = FinetuneDataset(test_df)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4, collate_fn=collate_fn
    )

    if os.path.exists(os.path.join(output_model_dir, f"backbone_after_finetune_{mode}.pt")):
        logger.info("Already exist, skipping training")
        test_ndcg = evalutate_model(
            torch.load(os.path.join(output_model_dir, f"backbone_after_finetune_{mode}.pt")), test_dataset, device
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

    prev_test_loss = None
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_epoch_loss = 0.0
        train_batches = 0

        for batch in tqdm(train_loader):
            batch = move_to_device(batch, device)
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
            train_batches += 1
        scheduler_warmup.step()

        avg_train_loss = train_epoch_loss / train_batches
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}")

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
        if prev_test_loss is None or avg_test_loss < prev_test_loss:
            prev_test_loss = avg_test_loss
            with torch.no_grad():
                torch.save(backbone, os.path.join(output_model_dir, f"backbone_after_finetune_{mode}.pt"))
        else:
            break

    test_ndcg = evalutate_model(
        torch.load(os.path.join(output_model_dir, f"backbone_after_finetune_{mode}.pt")), test_dataset, device
    )
    logger.info(f"Final Test NDCG: {test_ndcg:.6f}")
    return test_ndcg


if __name__ == "__main__":
    set_deterministic()
    # Command line argument setup
    parser = argparse.ArgumentParser(description="Training FinetuneModel")

    # Data
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)

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

    args = parser.parse_args()

    # Logger setup
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Starting FinetuneModel training")
    logger.info(f"Arguments: {args}")

    # Data loading
    logger.info(f"Loading training data from {args.train_data}")
    train_df = pl.read_parquet(args.train_data)

    logger.info(f"Loading test data from {args.test_data}")
    test_df = pl.read_parquet(args.test_data)

    result = dict()
    for init in ["frozen", "random", "twhin"]:
        backbone = torch.load(os.path.join(args.output_model_dir, f"backbone_after_pretrain_{init}.pt"))
        if init == "frozen":
            backbone.item_embeddings.weight.requires_grad = False

        result[init + "_ndcg@10"] = train_finetune_model(
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
        )

        logger.info(f"Training {init} completed successfully")

    with open(os.path.join(args.output_log_dir, "result.json"), "w") as file:
        json.dump(result, file, indent=4)

    logger.info(f"Results saved in directory: {args.output_log_dir}")
