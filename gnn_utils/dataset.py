import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class TwhinDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.row(idx)
        user_id, product_id, action_type = row[0], row[1], row[2]

        return {
            "users": torch.tensor(user_id, dtype=torch.long),
            "items": torch.tensor(product_id, dtype=torch.long),
            "relations": torch.tensor(action_type, dtype=torch.long),
        }


class PretrainDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.row(idx)
        product_id, product_names, action_type, candidate, candidate_names = row[0], row[1], row[3], row[4], row[5]

        return {
            "product_id": torch.tensor(product_id, dtype=torch.long),
            "product_names": {
                "ids": torch.tensor(product_names["ids"], dtype=torch.long),
                "lengths": torch.tensor(product_names["lengths"], dtype=torch.long),
            },
            "action_type": torch.tensor(action_type, dtype=torch.long),
            "candidate": torch.tensor(candidate, dtype=torch.long),
            "candidate_names": {
                "ids": torch.tensor(candidate_names["ids"], dtype=torch.long),
                "lengths": torch.tensor(candidate_names["lengths"], dtype=torch.long),
            },
        }


class FinetuneDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.row(idx)
        product_id, product_names, action_type, candidates, candidates_mask, candidate_names = (
            row[0],
            row[1],
            row[3],
            row[4],
            row[5],
            row[6],
        )

        return {
            "product_id": torch.tensor(product_id, dtype=torch.long),
            "product_names": {
                "ids": torch.tensor(product_names["ids"], dtype=torch.long),
                "lengths": torch.tensor(product_names["lengths"], dtype=torch.long),
            },
            "action_type": torch.tensor(action_type, dtype=torch.long),
            "candidate": torch.tensor(candidates, dtype=torch.long),
            "candidate_mask": torch.tensor(candidates_mask, dtype=torch.long),
            "candidate_names": {
                "ids": torch.tensor(candidate_names["ids"], dtype=torch.long),
                "lengths": torch.tensor(candidate_names["lengths"], dtype=torch.long),
            },
        }


def pretrain_collate_fn(batch):
    return {
        "product_id": pad_sequence([item["product_id"] for item in batch], batch_first=True, padding_value=0).to(
            torch.long
        ),
        "product_names": {
            "ids": torch.cat([item["product_names"]["ids"] for item in batch]),
            "lengths": torch.cat([item["product_names"]["lengths"] for item in batch]),
        },
        "action_type": pad_sequence([item["action_type"] for item in batch], batch_first=True, padding_value=0).to(
            torch.long
        ),
        "candidate": torch.cat([item["candidate"] for item in batch]),
        "candidate_names": {
            "ids": torch.cat([item["candidate_names"]["ids"] for item in batch]),
            "lengths": torch.cat([item["candidate_names"]["lengths"] for item in batch]),
        },
    }


def finetune_collate_fn(batch):
    return {
        "product_id": pad_sequence([item["product_id"] for item in batch], batch_first=True, padding_value=0).to(
            torch.long
        ),
        "product_names": {
            "ids": torch.cat([item["product_names"]["ids"] for item in batch]),
            "lengths": torch.cat([item["product_names"]["lengths"] for item in batch]),
        },
        "action_type": pad_sequence([item["action_type"] for item in batch], batch_first=True, padding_value=0).to(
            torch.long
        ),
        "candidate": torch.cat([item["candidate"] for item in batch]),
        "candidate_mask": torch.cat([item["candidate_mask"] for item in batch]),
        "candidate_lengths": torch.tensor([len(item["candidate"]) for item in batch], dtype=torch.long),
        "candidate_names": {
            "ids": torch.cat([item["candidate_names"]["ids"] for item in batch]),
            "lengths": torch.cat([item["candidate_names"]["lengths"] for item in batch]),
        },
    }
