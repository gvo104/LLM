import logging
from pathlib import Path

_LOGGER_NAME = "llm_lab.pipeline"


def configure_logging(log_path: str | Path) -> logging.Logger:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s request_id=%(request_id)s stage=%(stage)s %(message)s"
    )
    for handler in (logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, request_id: str, stage: str, message: str) -> None:
    logger.info(message, extra={"request_id": request_id, "stage": stage})
