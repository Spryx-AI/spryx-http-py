"""Custom logging configuration for spryx_http."""

import logging
import sys


def get_logger(name: str = "spryx_http") -> logging.Logger:
    """Get a configured logger for spryx_http.

    Args:
        name: Logger name, defaults to 'spryx_http'

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)

        # Create formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Set default level to INFO (can be overridden by environment)
        logger.setLevel(logging.INFO)

        # Prevent propagation to avoid duplicate logs
        logger.propagate = False

    return logger


# Create default logger instance
logger = get_logger()
