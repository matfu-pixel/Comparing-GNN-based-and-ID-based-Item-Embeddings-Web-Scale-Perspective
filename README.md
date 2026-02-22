# Comparing GNN-based and ID-based Item Embeddings on Yandex Lavka dataset

## Overview

This repository implements a transformer-based ranking model over user-item interaction history for [Yandex Lavka](https://lavka.yandex.ru/), a large-scale e-commerce service. We compare two ways of passing items to the transformer: either as pretrained [TwHIN](https://arxiv.org/pdf/2202.05387) item embeddings or as fully trainable item ID embeddings learned from scratch (without pretraining).

The question we are trying to answer is: *Is it worth training GNN item embeddings to represent items in sequential recommendation transformer-based models?*

## Data

We publish the anonymized logs from Yandex Lavka utilized in this project. These logs, sampled over a one-year period, encompass 15 million user-item interactions from 3,315 users and 25,833 items.

There are two options to get the training data:

1. For your convenience, we have tracked the data with [DVC](https://dvc.org/), so that you can download everything you need with a single command. For that, however, you need to follow [this guide](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended) to obtain Google OAuth client credentials. Find more details about authentication [here](https://github.com/treeverse/dvc/issues/10516#issuecomment-2289652067).
2. Alternatively, if you prefer not to use DVC, you can simply download the raw data from [Zenodo](https://zenodo.org/records/15529491?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImE2YjMzYTg3LTc1OGItNDlhZS1hMTc5LTQyNjRlYjFiYzcwNSIsImRhdGEiOnt9LCJyYW5kb20iOiJlN2M5YjA5MmE2MjI4MDAxOWZjN2UyODhjYTM0ODk3YyJ9.6puVZtP2dmS4bis00RmmeoERl0jGyzuX0rMmNna7wULDxqgB45quLjSXFG2iakyyRW2G7bajty1ElD0gVlkofw).

Each user-item interaction (a dataset row) has the following fields:

- `user_id`: Unique user identifier
- `timestamp`: Interaction timestamp
- `product_id`: Unique item identifier
- `request_id`: Recommendation request identifier (items are ranked within requests)
- `action_type`: Type of user interaction (view, click, add to cart, purchase)
- `product_name`: Tokenized item name

## Installation

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to handle dependencies. Please install it before moving forward:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository:
```bash
git clone git@github.com:matfu-pixel/Comparing-GNN-based-and-ID-based-Item-Embeddings-Web-Scale-Perspective.git gnn_utils && cd gnn_utils
```

Initialize the virtual environment:

```bash
uv python install 3.12 && uv venv --python 3.12
```

You have four options to install the dependencies:

1. Install minimal dependencies:
```bash
uv sync
```
2. Include DVC if you intend to use it to pull the training data:
```bash
uv sync --extra dvc
```
3. Optionally, install [MLflow](https://mlflow.org/) to track the training progress:
```bash
uv sync --extra mlflow
```
4. Include both MLflow and DVC:
```bash
uv sync --extra dvc --extra mlflow
```

**Note:** Make sure the above command installed a PyTorch version compatible with your CUDA version. Otherwise, override it with `uv pip install torch <YOUR REQUIRED DISTRIBUTION>`.

## Downloading data

#### 1. Pull the data with DVC

If you configured DVC successfully (see the Data section above), you should end up with `.dvc/config.local` or `.dvc/config` containing:

```ini
['remote "storage"']
    gdrive_client_id = <YOUR CLIENT ID>
    gdrive_client_secret = <YOUR SECRET>
```

Pull the raw data and the preprocessed datasets with a single command:
```bash
uv run dvc pull
```

#### 2. Or download it manually:

If you downloaded the data manually from [Zenodo](https://zenodo.org/records/15529491?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImE2YjMzYTg3LTc1OGItNDlhZS1hMTc5LTQyNjRlYjFiYzcwNSIsImRhdGEiOnt9LCJyYW5kb20iOiJlN2M5YjA5MmE2MjI4MDAxOWZjN2UyODhjYTM0ODk3YyJ9.6puVZtP2dmS4bis00RmmeoERl0jGyzuX0rMmNna7wULDxqgB45quLjSXFG2iakyyRW2G7bajty1ElD0gVlkofw), save it as `./data/dataset.parquet`.

## Data processing

If you pulled the data with DVC, the preprocessed datasets should already be in `./data`; you can skip this step.

Run the preprocessing step (takes ~10 minutes) with:
```bash
./run_pipeline.sh --prepare
```

## Training

Train the TwHIN item embeddings (takes ~1 minute):
```bash
./run_pipeline.sh --train-twhin
```

The ranking model training is divided into two stages: `pretraining` and `fine-tuning`, as described in [this paper](https://arxiv.org/pdf/2310.03481).

Run `pretraining` stage (takes ~40 minutes):
```bash
./run_pipeline.sh --train-pretrain
```

Run `fine-tuning` stage (takes ~20 minutes):
```bash
./run_pipeline.sh --train-finetune
```

If you installed MLflow, you can track the training progress online:
```bash
uv run mlflow server --port 5010
```

## Evaluation

The recommendation quality is evaluated using nDCG per `request_id`:

- **Positive examples**: Cart additions
- **Negative examples**: Viewed items

The final metric is the average nDCG across all request IDs. 

Evaluation is performed after the `fine-tuning` stage. You can find the evaluation results in `./logs`.

To average results over `K` runs:
```bash
./run_multiple_times.sh K
```
The averaged statistics will be saved as `./logs/averaged.json`.

## Conclusions

The results are averaged across 10 runs. 

| Embedding Initialization | End-to-end fine-tuning | nDCG@5 | nDCG@10 | nDCG@20 |
|:------------------------|:-------------|-------:|--------:|--------:|
| Random | ✓ | 0.333 | 0.406 | 0.456 |
| From TwHIN | ✗ | <u>0.337</u> | <u>0.409</u> | <u>0.457</u> |
| From TwHIN | ✓ | **0.342** | **0.415** | **0.464** |

Pretrained TwHIN item embeddings, frozen during ranking model training, demonstrate slightly superior performance compared to ID-embeddings trained end-to-end from random initialization on this dataset. Further improvements are observed when fine-tuning the pretrained TwHIN item embeddings.

However, internal experiments with significantly larger datasets indicate that end-to-end trained item ID embeddings substantially outperform frozen TwHIN embeddings. Combining both approaches yields negligible improvements over using only end-to-end trained item ID embeddings.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
