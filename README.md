# Comparing GNN-based and ID-based Item Embeddings: Web-Scale Perspective 

## Overview

This repository implements a recommendation system for Online Food Delivery Services. The system is designed to rank food products based on user interaction data, with the goal of optimizing recommendations to increase the likelihood of users adding products to their shopping carts.

## Repository Structure

```
utils/
    - losses.py        # Custom loss functions
    - models.py        # Model architectures
.gitignore
preprocess.py          # Data preprocessing script
README.md              # Repository documentation
requirements.txt       # Project dependencies
run_pipeline.sh        # Pipeline execution script
train_finetune.py      # Model fine-tuning
train_pretrain.py      # Model pre-training
train_twhin.py         # Advanced training approach
```


## Installation

Clone the repository:
```bash
git clone git@github.com:matfu-pixel/Comparing-GNN-based-and-ID-based-Item-Embeddings-Web-Scale-Perspective.git gnn_utils
cd gnn_utils
```

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to handle dependencies. Please install it before moving forward:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To set up the project:

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv sync --dev
```

Note: Currently, this installs PyTorch for CUDA 12.8. If this does not work for you, or if you want another CUDA build, override it with `uv pip install torch <YOUR REQUIRED DISTRIBUTION>`.


## Data

### Download data 

We use [DVC](https://dvc.org/) to track the data and model weights. The Google Drive is used as a remote storage. Follow [this instruction](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended) to obtain the required credentials. You will end up with `.dvc/config.local` file containing:
```
['remote "storage"']
    gdrive_client_id = <YOUR CLIENT ID>
    gdrive_client_secret = <YOUR SECRET>
```
Pull the data and model checkpoints:
```bash
dvc pull
dvc repro
```

Alternatively, you can simply download the [raw data](https://zenodo.org/records/15529491?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImE2YjMzYTg3LTc1OGItNDlhZS1hMTc5LTQyNjRlYjFiYzcwNSIsImRhdGEiOnt9LCJyYW5kb20iOiJlN2M5YjA5MmE2MjI4MDAxOWZjN2UyODhjYTM0ODk3YyJ9.6puVZtP2dmS4bis00RmmeoERl0jGyzuX0rMmNna7wULDxqgB45quLjSXFG2iakyyRW2G7bajty1ElD0gVlkofw) and put it into `data/dataset.parquet`.


## Data Description

The model is trained on user interaction data from an Online Food Delivery Service with the following structure:

- `user_id`: Unique identifier for users
- `timestamp`: Time when the interaction occurred
- `product_id`: Unique identifier for food products
- `request_id`: Identifier for recommendation requests (recommendations are grouped by request)
- `action_type`: Type of user interaction (view, click, add to cart, purchase)
- `product_name`: Tokenized product name

First, download the data from https://zenodo.org/records/15529491?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImE2YjMzYTg3LTc1OGItNDlhZS1hMTc5LTQyNjRlYjFiYzcwNSIsImRhdGEiOnt9LCJyYW5kb20iOiJlN2M5YjA5MmE2MjI4MDAxOWZjN2UyODhjYTM0ODk3YyJ9.6puVZtP2dmS4bis00RmmeoERl0jGyzuX0rMmNna7wULDxqgB45quLjSXFG2iakyyRW2G7bajty1ElD0gVlkofw, then place it in the `/data` directory and make sure the file is named `dataset.parquet`.


## Usage

To run the complete pipeline (data preprocessing, training, and evaluation):

```bash
./run_pipeline.sh
```

This script will:

1. Preprocess the input data
2. Train recommendation models through the multi-stage approach
3. Evaluate model performance
4. Save outputs to the designated directories

## Evaluation

The recommendation quality is evaluated using NDCG@10 (Normalized Discounted Cumulative Gain at 10) per request_id:

- **Positive examples**: Products added to cart
- **Negative examples**: Products only viewed
- The final metric is averaged across all request_ids


## Output

After running the pipeline, you'll find:

- Model checkpoints in `./tmp/models/`
- Processed datasets in `./tmp/data/`
- Evaluation results and logs in `./tmp/logs/`


## License

[Specify license information here]

## Contributors

[List contributors here]

