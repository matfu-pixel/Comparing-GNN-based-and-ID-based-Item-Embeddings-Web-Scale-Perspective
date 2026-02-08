# Comparing GNN-based and ID-based Item Embeddings on Yandex Lavka dataset

## Overview

This repository implements a transformer-based ranking model over user-item interaction history for [Yandex Lavka](https://lavka.yandex.ru/), a large-scale e-commerce service. We compare two ways of passing items to the transformer: either as pretrained [TwHIN](https://arxiv.org/pdf/2202.05387) item embeddings or as fully trainable embeddings learned from scratch (without pretraining).

The question we are trying to answer is: `Is it worth training GNN item embeddings to represent items in sequential recommendation transformer-based models?`

## Data

We publish the anonymized logs from Yandex Lavka used in this project. There are two options to get the training data:

1. For your convenience, we have tracked the data with [DVC](https://dvc.org/), so that you can download everything you need with a single command. For that, however, you need to follow [this guide](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended) to obtain Google OAuth client credentials. Find more details about authentication [here](https://github.com/treeverse/dvc/issues/10516#issuecomment-2289652067).
2. Alternatively, if you prefer not to use DVC, you can download the raw data from [Zenodo](https://zenodo.org/records/15291186?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjY5Y2NlNzA5LWRjOGMtNGZhMy05OTY1LWFhNGFmMThhMTk5YSIsImRhdGEiOnt9LCJyYW5kb20iOiI2OGQxMjBmMTE3YjQ0YTBlNWFmMjZkYjU3NjU0NTQ5MyJ9.cZJTpykinsmAMPdJzfdKK3Dg5l906oVNJYJZ3ayhbVigU_vAOU7NVTf_CCAMit4MdNpvl5CH4T4Z6MIBUKBGhg).

The model is trained on user interaction data from Yandex Lavka:

- `user_id`: Unique user identifier
- `timestamp`: Interaction timestamp
- `product_id`: Unique item identifier
- `request_id`: Recommendation request identifier (recommendations are grouped by requests)
- `action_type`: Type of user interaction (view, click, add to cart, purchase)
- `product_name`: Tokenized item name

## Installation

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to handle dependencies. Please install it before moving forward:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository:
```bash
git clone git@github.com:matfu-pixel/Comparing-GNN-based-and-ID-based-Item-Embeddings-Web-Scale-Perspective.git gnn_utils
cd gnn_utils
```

Initialize the virtual environment:

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
```

You have four options to install the dependencies:

1. Install minimal dependencies:
```bash
uv sync
```
2. Include `dvc` if you intend to use it to pull the training data:
```bash
uv sync --extra dvc
```
3. Optionally, install [mlflow](https://mlflow.org/) to track the training progress:
```bash
uv sync --extra mlflow
```
4. Include both mlflow and dvc:
```bash
uv sync --extra dvc --extra mlflow
```

*Note:* Make sure the above command installed a PyTorch version compatible with your CUDA version. Otherwise, override it with `uv pip install torch <YOUR REQUIRED DISTRIBUTION>`.

## Downloading data

#### 1. Pull the data with DVC

If you configured `dvc` successfully (see the Data section), you should end up with `.dvc/config.local` or `.dvc/config` file containing:
```
['remote "storage"']
    gdrive_client_id = <YOUR CLIENT ID>
    gdrive_client_secret = <YOUR SECRET>
```

Pull the raw data and the preprocessed datasets with a single command:
```bash
dvc pull
```

#### 2. Or download it manually:

If you downloaded the data manually from [Zenodo](https://zenodo.org/records/15291186?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjY5Y2NlNzA5LWRjOGMtNGZhMy05OTY1LWFhNGFmMThhMTk5YSIsImRhdGEiOnt9LCJyYW5kb20iOiI2OGQxMjBmMTE3YjQ0YTBlNWFmMjZkYjU3NjU0NTQ5MyJ9.cZJTpykinsmAMPdJzfdKK3Dg5l906oVNJYJZ3ayhbVigU_vAOU7NVTf_CCAMit4MdNpvl5CH4T4Z6MIBUKBGhg), save it as `./data/dataset.parquet`.

## Data processing

If you pulled the data with `dvc`, the preprocessed datasets should be already in `./data`; you can skip this step.

Run the preprocessing step (takes ~10 minutes) with:
```bash
./run_pipeline.sh --prepare
```

## Training

Train the TwHIN item embeddings (takes ~1 minute):
```bash
./run_pipeline.sh --train-twhin
```

The training process is split into two stages: `pretraining` and `fine-tuning`, as described in [this paper](https://arxiv.org/pdf/2310.03481).

Run `pretraining` stage:
```bash
./run_pipeline.sh --train-pretrain
```

Run `fine-tuning` stage:
```bash
./run_pipeline.sh --train-finetune
```

## Evaluation

The recommendation quality is evaluated using nDCG@10 per `request_id`:

- **Positive examples**: Cart additions
- **Negative examples**: Viewed items

The final metric is the average nDCG@10 across all `request_id`s. 

Evaluation is performed after the `fine-tuning` stage. You can find the evaluation results in `./logs`.

## Conclusions

Overall, the results suggest that, at web scale, a transformer trained on rich user–item interaction histories can learn effective item representations from scratch, and the main advantages of pretrained GNN-based item embeddings appear when training data for the transformer is scarce.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
