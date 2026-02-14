import logging


try:
    import mlflow  # type: ignore
except ImportError:
    from . import noop_mlflow as mlflow  # noqa: F401

logging.getLogger("alembic").setLevel(logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.WARNING)
