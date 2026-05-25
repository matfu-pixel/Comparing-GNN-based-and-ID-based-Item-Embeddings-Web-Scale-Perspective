import importlib
import logging
from typing import Any


try:
    import mlflow as _mlflow
except ImportError:
    mlflow: Any = importlib.import_module("gnn_utils.tracking.noop_mlflow")
else:
    mlflow: Any = _mlflow

logging.getLogger("alembic").setLevel(logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.WARNING)
