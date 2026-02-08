from contextlib import contextmanager


@contextmanager
def start_run(*args, **kwargs):
    yield


def set_experiment(*args, **kwargs):
    pass


def log_metric(*args, **kwargs):
    pass


def log_metrics(*args, **kwargs):
    pass


def log_param(*args, **kwargs):
    pass


def log_params(*args, **kwargs):
    pass


def log_artifact(*args, **kwargs):
    pass


def log_artifacts(*args, **kwargs):
    pass


class _NoOpConfig:
    def enable_system_metrics_logging(self, *args, **kwargs):
        pass

    def set_system_metrics_sampling_interval(self, *args, **kwargs):
        pass


config = _NoOpConfig()
