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


## Data Description

The system is trained on user interaction data from an Online Food Delivery Service with the following structure:

- `user_id`: Unique identifier for users
- `timestamp`: Time when the interaction occurred
- `product_id`: Unique identifier for food products
- `request_id`: Identifier for recommendation requests (recommendations are grouped by request)
- `action_type`: Type of user interaction (view, click, add to cart, purchase)

**Note:** The actual dataset will be made available after the anonymous review process is completed to maintain the confidentiality of the company.


## Installation

To set up the project:

```bash
git clone [repository-url]
cd [repository-name]
pip install -r requirements.txt
```


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

