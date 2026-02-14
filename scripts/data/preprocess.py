import argparse
import logging
import os
from abc import ABC, abstractmethod
from collections import deque
from itertools import chain
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import polars as pl


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ensure_sorted_by_timestamp(group: Iterable[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
    """
    Verify that events are sorted by timestamp and yield each event.

    Args:
        group: Iterable of event dictionaries, expected to be sorted by timestamp

    Yields:
        Event dictionaries in original order

    Raises:
        AssertionError: If events are not sorted by timestamp
    """
    prev_timestamp: Optional[int] = None
    for row in group:
        if prev_timestamp is not None and prev_timestamp > row["timestamp"]:
            raise AssertionError("Events must be sorted by timestamp")
        prev_timestamp = row["timestamp"]
        yield row


class LaggedQueue:
    """
    Queue that maintains events within a specified time lag window.

    This queue maintains two internal deques:
    - _lagged_deque: Contains events that are older than the lag time
    - _fresh_deque: Contains more recent events that haven't yet passed the lag threshold
    """

    def __init__(self, lag: int, capacity: int):
        """
        Initialize a new LaggedQueue.

        Args:
            lag: Time lag in seconds
            capacity: Maximum capacity of the lagged queue
        """
        self._lag: int = lag
        self._capacity: int = capacity
        self._lagged_deque: deque = deque(maxlen=capacity)
        self._fresh_deque: deque = deque()

    def __len__(self) -> int:
        """Return the number of events in the lagged queue."""
        return len(self._lagged_deque)

    def get(self) -> List[Dict[str, Any]]:
        """Return all events in the lagged queue."""
        return list(self._lagged_deque)

    def update(self, timestamp: int) -> None:
        """
        Update the lagged queue based on the current timestamp.

        This moves events from the fresh deque to the lagged deque
        if they're older than the lag threshold.

        Args:
            timestamp: Current timestamp to compare against
        """
        while self._fresh_deque and (timestamp - self._fresh_deque[0]["timestamp"]) > self._lag:
            self._lagged_deque.append(self._fresh_deque.popleft())

    def push(self, event: Dict[str, Any]) -> None:
        """
        Add a new event to the queue and update internal state.

        Args:
            event: Event dictionary containing at least a 'timestamp' key
        """
        self._fresh_deque.append(event)
        self.update(event["timestamp"])


class ActionType:
    """Constants for action types to avoid string literals throughout the code."""

    VIEW = "AT_View"
    CLICK = "AT_Click"
    CART_UPDATE = "AT_CartUpdate"
    PURCHASE = "AT_Purchase"
    TARGET = "target"

    @classmethod
    def get_standard_types(cls) -> List[str]:
        """Return list of standard action types."""
        return [cls.VIEW, cls.CLICK, cls.CART_UPDATE, cls.PURCHASE]


class Reducer(ABC):
    """
    Base class for data reducers that process groups of events.
    """

    def __init__(self, min_length: int, max_length: int, lag: int, timesplit: int, result_schema: Dict[str, Any]):
        """
        Initialize a new Reducer.

        Args:
            min_length: Minimum number of events required before processing
            max_length: Maximum number of events to keep in history
            lag: Time lag in seconds for the LaggedQueue
            timesplit: Timestamp to split data into training and validation sets
            result_schema: Schema for the result DataFrame
        """
        self._min_length: int = min_length
        self._max_length: int = max_length
        self._lag: int = lag
        self._timesplit: int = timesplit
        self._result_schema: Dict[str, Any] = result_schema

    @abstractmethod
    def __call__(self, group: pl.DataFrame) -> pl.DataFrame:
        """
        Process a group of events and return a reduced DataFrame.

        Args:
            group: DataFrame containing events for a single group (e.g., user)

        Returns:
            Processed DataFrame with the specified result schema
        """
        pass


class PretrainReducer(Reducer):
    """
    Reducer for preprocessing data for pretraining tasks.

    This reducer creates sequences of user interactions and targets
    for pretraining recommendation models.
    """

    def __call__(self, group: pl.DataFrame) -> pl.DataFrame:
        """
        Process a group of events for pretraining.

        Args:
            group: DataFrame containing events for a single user

        Returns:
            Processed DataFrame with interaction sequences and targets
        """
        res_rows: List[Dict[str, Any]] = []
        history = LaggedQueue(lag=self._lag, capacity=self._max_length)

        for row in ensure_sorted_by_timestamp(group.to_dicts()):
            if row["action_type"] in {ActionType.CART_UPDATE}:
                history.update(row["timestamp"])

                if len(history) >= self._min_length:
                    list_of_dicts = history.get()
                    res_row: Dict[str, List] = {}

                    for key in ["product_id", "timestamp", "action_type"]:
                        res_row[key] = [sample[key] for sample in list_of_dicts]
                    res_row["product_names"] = {
                        "ids": list(chain.from_iterable([sample["product_name"] for sample in list_of_dicts])),
                        "lengths": [len(sample["product_name"]) for sample in list_of_dicts],
                    }

                    # Determine train/test split
                    res_row["is_valid"] = [row["timestamp"] > self._timesplit]
                    res_row["candidate"] = [row["product_id"]]
                    res_row["candidate_names"] = {"ids": row["product_name"], "lengths": [len(row["product_name"])]}
                    res_rows.append(res_row)

                history.push(row)
            elif row["action_type"] in {ActionType.VIEW, ActionType.CLICK, ActionType.PURCHASE}:
                history.push(row)
            else:
                raise ValueError(f"Unknown action type: {row['action_type']}")

        if not res_rows:
            return pl.DataFrame(schema=self._result_schema)

        return pl.DataFrame(res_rows, schema=self._result_schema)


class FinetuneReducer(Reducer):
    """
    Reducer for preprocessing data for fine-tuning tasks.

    This reducer creates sequences of user interactions and targets
    for fine-tuning recommendation models.
    """

    def __call__(self, group: pl.DataFrame) -> pl.DataFrame:
        """
        Process a group of events for fine-tuning.

        Args:
            group: DataFrame containing events for a single user

        Returns:
            Processed DataFrame with interaction sequences and candidates
        """
        res_rows: List[Dict[str, Any]] = []
        history = LaggedQueue(lag=self._lag, capacity=self._max_length)

        for row in ensure_sorted_by_timestamp(group.to_dicts()):
            if row["action_type"] == ActionType.TARGET:
                history.update(row["timestamp"])

                if len(history) >= self._min_length:
                    list_of_dicts = history.get()
                    res_row: Dict[str, Any] = {}

                    for key in ["product_id", "timestamp", "action_type"]:
                        res_row[key] = [sample[key] for sample in list_of_dicts]
                    res_row["product_names"] = {
                        "ids": list(chain.from_iterable([sample["product_name"] for sample in list_of_dicts])),
                        "lengths": [len(sample["product_name"]) for sample in list_of_dicts],
                    }

                    # Determine train/test split
                    res_row["is_valid"] = [row["timestamp"] > self._timesplit]
                    res_row["candidates"] = row["candidates"]
                    res_row["candidates_mask"] = row["candidates_mask"]
                    res_row["candidate_names"] = {
                        "ids": list(chain.from_iterable(row["product_name_list"])),
                        "lengths": [len(sample) for sample in row["product_name_list"]],
                    }
                    res_rows.append(res_row)
            elif row["action_type"] in {ActionType.VIEW, ActionType.CLICK, ActionType.CART_UPDATE, ActionType.PURCHASE}:
                history.push(row)
            else:
                raise ValueError(f"Unknown action type: {row['action_type']}")

        if not res_rows:
            return pl.DataFrame(schema=self._result_schema)
        return pl.DataFrame(res_rows, schema=self._result_schema)


class Preprocessor:
    """
    Preprocessor for user interaction data.

    This class handles preprocessing of user interaction data for different modeling tasks:
    - twhin: For graph-based modeling
    - pretrain: For pretraining recommendation models
    - finetune: For fine-tuning recommendation models
    """

    # Map action types to integer IDs for models
    mapping_action_types = {
        ActionType.CLICK: 0,
        ActionType.CART_UPDATE: 1,
        ActionType.PURCHASE: 2,
        ActionType.VIEW: 3,
    }

    def __init__(self, data: pl.LazyFrame) -> None:
        """
        Initialize a new Preprocessor.

        Args:
            data: LazyFrame containing user interaction data
        """
        self._data: pl.LazyFrame = data

    def preprocess_data(self) -> None:
        """
        Preprocess the data by mapping product and user IDs to integers.

        This modifies the internal data representation by replacing
        product and user IDs with sequential integer IDs.
        """
        unique_product_ids = self._data.select(pl.col("product_id")).unique().collect().to_series().to_list()
        mapping_product_ids = {val: idx for idx, val in enumerate(sorted(unique_product_ids))}
        self._data = self._data.with_columns(
            pl.col("product_id").replace(mapping_product_ids).add(1).alias("product_id")  # add 1 for padding token
        )
        logger.info(f"Preprocessed {len(mapping_product_ids)} unique product IDs")

        unique_user_ids = self._data.select(pl.col("user_id")).unique().collect().to_series().to_list()
        mapping_user_ids = {val: idx for idx, val in enumerate(sorted(unique_user_ids))}
        self._data = self._data.with_columns(pl.col("user_id").replace(mapping_user_ids).alias("user_id"))
        logger.info(f"Preprocessed {len(mapping_user_ids)} unique user IDs")

    def preprocess_twhin_data(self, time_split: int) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Preprocess data for twhin (graph-based) modeling.

        Args:
            time_split: Timestamp to split data into training and test sets

        Returns:
            Tuple of (train_data, test_data) DataFrames
        """
        result = (
            self._data.select(pl.col("user_id", "product_id", "timestamp", "action_type"))
            .filter(pl.col("action_type").is_in([ActionType.CLICK, ActionType.CART_UPDATE, ActionType.PURCHASE]))
            .with_columns((pl.col("timestamp") > time_split).alias("is_valid"))
            .select(pl.col("user_id", "product_id", "action_type", "is_valid"))
            .with_columns(pl.col("action_type").replace(self.mapping_action_types).cast(pl.Int8).alias("action_type"))
        )

        # Split into train and test based on is_valid
        train_data, test_data = pl.collect_all(
            [result.filter(~pl.col("is_valid")).drop("is_valid"), result.filter(pl.col("is_valid")).drop("is_valid")]
        )

        logger.info(f"Preprocessed twhin data: {len(train_data)} train rows, {len(test_data)} test rows")
        return train_data, test_data

    def preprocess_pretrain_data(
        self, time_split: int, min_length: int = 5, max_length: int = 256, lag_seconds: int = 86400
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Preprocess data for pretraining recommendation models.

        Args:
            time_split: Timestamp to split data into training and test sets
            min_length: Minimum number of events required before processing
            max_length: Maximum number of events to keep in history
            lag_seconds: Time lag in seconds for the LaggedQueue

        Returns:
            Tuple of (train_data, test_data) DataFrames
        """
        RESULT_SCHEMA = {
            "product_id": pl.List(pl.UInt64),
            "product_names": pl.Struct({"ids": pl.List(pl.Int64), "lengths": pl.List(pl.Int64)}),
            "timestamp": pl.List(pl.Int64),
            "action_type": pl.List(pl.Utf8),
            "is_valid": pl.List(pl.Boolean),
            "candidate": pl.List(pl.UInt64),
            "candidate_names": pl.Struct({"ids": pl.List(pl.Int64), "lengths": pl.List(pl.Int64)}),
        }
        prepared_data = self._data.select(
            pl.col("user_id", "product_id", "timestamp", "action_type", "product_name")
        ).sort(["user_id", "timestamp"])

        reducer = PretrainReducer(
            min_length=min_length,
            max_length=max_length,
            lag=lag_seconds,
            timesplit=time_split,
            result_schema=RESULT_SCHEMA,
        )

        result = (
            prepared_data.group_by("user_id")
            .map_groups(reducer, schema=RESULT_SCHEMA)
            .with_columns(
                pl.col("action_type").list.eval(
                    pl.element().map_elements(
                        lambda s: self.mapping_action_types[s],
                        return_dtype=pl.Int8,
                    )
                )
            )
        )

        # Split into train and test based on is_valid
        train_lf = result.filter(~pl.col("is_valid").list.all()).drop("is_valid")
        test_lf = result.filter(pl.col("is_valid").list.all()).drop("is_valid")

        train_data, test_data = pl.collect_all([train_lf, test_lf])

        logger.info(f"Preprocessed pretrain data: {len(train_data)} train rows, {len(test_data)} test rows")
        return train_data, test_data

    def _prepare_targets(self) -> pl.LazyFrame:
        """
        Prepare target data for fine-tuning.

        This method extracts and transforms target events from raw data.

        Returns:
            LazyFrame with prepared target data
        """
        return (
            self._data.select(pl.col("user_id", "product_id", "timestamp", "action_type", "request_id", "product_name"))
            .filter(pl.col("action_type").is_in([ActionType.VIEW, ActionType.CART_UPDATE]))
            .with_columns(target=pl.when(pl.col("action_type") == ActionType.VIEW).then(0).otherwise(1))
            .drop("action_type")
            .group_by(["product_id", "request_id"])
            .agg(
                [
                    pl.col("user_id").min(),
                    pl.col("timestamp").min(),
                    pl.col("target").max(),
                    pl.col("product_name").first(),
                ]
            )
            .group_by("request_id")
            .agg(
                pl.col("user_id").min(),
                pl.col("timestamp").min(),
                pl.col("target").alias("candidates_mask"),
                pl.col("target").sum().alias("postive_candidates_count"),
                pl.col("target").count().alias("candidates_count"),
                pl.col("product_id").alias("candidates"),
                pl.col("product_name").alias("product_name_list"),
            )
            .with_columns(pl.lit(ActionType.TARGET).alias("action_type"))
            .filter((pl.col("postive_candidates_count") > 0) & (pl.col("candidates_count") > 1))
            .drop(["postive_candidates_count", "candidates_count"])
        )

    def preprocess_finetune_data(
        self, time_split: int, min_length: int = 5, max_length: int = 256, lag_seconds: int = 86400
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Preprocess data for fine-tuning recommendation models.

        Args:
            time_split: Timestamp to split data into training and test sets
            min_length: Minimum number of events required before processing
            max_length: Maximum number of events to keep in history
            lag_seconds: Time lag in seconds for the LaggedQueue

        Returns:
            Tuple of (train_data, test_data) DataFrames
        """
        RESULT_SCHEMA = {
            "product_id": pl.List(pl.UInt64),
            "product_names": pl.Struct({"ids": pl.List(pl.Int64), "lengths": pl.List(pl.Int64)}),
            "timestamp": pl.List(pl.Int64),
            "action_type": pl.List(pl.Utf8),
            "is_valid": pl.List(pl.Boolean),
            "candidates": pl.List(pl.UInt64),
            "candidates_mask": pl.List(pl.Boolean),
            "candidate_names": pl.Struct({"ids": pl.List(pl.Int64), "lengths": pl.List(pl.Int64)}),
        }

        # Prepare regular interaction data
        prepared_data = self._data.select(pl.col("user_id", "product_id", "timestamp", "action_type", "product_name"))

        # Prepare target data
        prepared_targets = self._prepare_targets()

        # Combine interaction data with targets and sort by user and timestamp
        prepared_data = pl.concat([prepared_data, prepared_targets], how="diagonal").sort(["user_id", "timestamp"])

        reducer = FinetuneReducer(
            min_length=min_length,
            max_length=max_length,
            lag=lag_seconds,
            timesplit=time_split,
            result_schema=RESULT_SCHEMA,
        )

        result = (
            prepared_data.group_by("user_id")
            .map_groups(reducer, schema=RESULT_SCHEMA)
            .with_columns(
                pl.col("action_type").list.eval(
                    pl.element().map_elements(
                        lambda s: self.mapping_action_types[s],
                        return_dtype=pl.Int8,
                    )
                )
            )
        )

        # Split into train and test based on is_valid
        train_lf = result.filter(~pl.col("is_valid").list.all()).drop("is_valid")
        test_lf = result.filter(pl.col("is_valid").list.all()).drop("is_valid")

        train_data, test_data = pl.collect_all([train_lf, test_lf])

        logger.info(f"Preprocessed finetune data: {len(train_data)} train rows, {len(test_data)} test rows")
        return train_data, test_data


def main(
    input_path: str,
    output_train_twhin_path: str,
    output_test_twhin_path: str,
    output_train_pretrain_path: str,
    output_test_pretrain_path: str,
    output_train_finetune_path: str,
    output_test_finetune_path: str,
    time_split: int,
    force: bool = False,
) -> None:
    """
    Process input data and save preprocessed datasets for different modeling tasks.

    Args:
        input_path: Path to input Parquet file
        output_train_twhin_path: Path to save training data for twhin
        output_test_twhin_path: Path to save test data for twhin
        output_train_pretrain_path: Path to save training data for pretrain
        output_test_pretrain_path: Path to save test data for pretrain
        output_train_finetune_path: Path to save training data for finetune
        output_test_finetune_path: Path to save test data for finetune
        time_split: Timestamp to split data into training and test sets
        force: Whether to overwrite existing files
    """
    # Load data
    logger.info(f"Loading data from {input_path}...")
    data = pl.scan_parquet(input_path)
    preprocessor = Preprocessor(data=data)
    preprocessor.preprocess_data()

    # Define preprocessing modes and output paths
    modes_and_paths = [
        ("twhin", (output_train_twhin_path, output_test_twhin_path)),
        ("pretrain", (output_train_pretrain_path, output_test_pretrain_path)),
        ("finetune", (output_train_finetune_path, output_test_finetune_path)),
    ]

    # Process each mode
    for mode, (output_train_path, output_test_path) in modes_and_paths:
        # Check if output files already exist
        if not force and os.path.exists(output_train_path) and os.path.exists(output_test_path):
            logger.warning(f"Files {output_train_path} and {output_test_path} already exist for {mode}. Skipping.")
            continue

        # Preprocess data
        logger.info(f"Processing data for {mode}...")
        train_data, test_data = getattr(preprocessor, f"preprocess_{mode}_data")(time_split=time_split)

        # Save results
        logger.info(f"Saving train data ({len(train_data)} rows) to {output_train_path} for {mode}...")
        train_data.write_parquet(output_train_path)

        logger.info(f"Saving test data ({len(test_data)} rows) to {output_test_path} for {mode}...")
        test_data.write_parquet(output_test_path)

        logger.info(f"Processing of {mode} completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event data preprocessing for recommendation models.")
    parser.add_argument("--input", type=str, required=True, help="Path to input Parquet file")
    parser.add_argument(
        "--output-train-twhin",
        type=str,
        default="train_data_twhin.parquet",
        help="Path to save training data for twhin",
    )
    parser.add_argument(
        "--output-test-twhin", type=str, default="test_data_twhin.parquet", help="Path to save test data for twhin"
    )
    parser.add_argument(
        "--output-train-pretrain",
        type=str,
        default="train_data_pretrain.parquet",
        help="Path to save training data for pretrain",
    )
    parser.add_argument(
        "--output-test-pretrain",
        type=str,
        default="test_data_pretrain.parquet",
        help="Path to save test data for pretrain",
    )
    parser.add_argument(
        "--output-train-finetune",
        type=str,
        default="train_data_finetune.parquet",
        help="Path to save training data for finetune",
    )
    parser.add_argument(
        "--output-test-finetune",
        type=str,
        default="test_data_finetune.parquet",
        help="Path to save test data for finetune",
    )
    parser.add_argument(
        "--time-split", type=int, default=1703451224, help="Time split for twhin, pretrain and finetune (timestamp)"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")

    args = parser.parse_args()
    main(
        args.input,
        args.output_train_twhin,
        args.output_test_twhin,
        args.output_train_pretrain,
        args.output_test_pretrain,
        args.output_train_finetune,
        args.output_test_finetune,
        args.time_split,
        args.force,
    )
