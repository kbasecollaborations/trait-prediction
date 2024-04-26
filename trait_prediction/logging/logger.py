"""Module that defines the logger configuration"""

from loguru import logger

config = {
    "handlers": [
        # Add a logger that is multiprocess-safe
        {"sink": "file.log", "enqueue": True},
    ],
    "extra": None,
}
logger.configure(**config)

logger.disable("trait_prediction")
