"""
File: logger_setup.py
Purpose: Configures a rotating file logger and a console handler for all
         MIRAGE modules. Import and call setup_logging() once per process.

Part of the audit codebase (diagnosis half of CURE).
"""

import logging
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(run_id: str | None = None, level: int = logging.INFO) -> str:
    """
    Initialise root logger with rotating file handler and console handler.

    Parameters
    ----------
    run_id : str or None
        UUID for this run. If None, a fresh UUID4 is generated.
    level : int
        Logging level (default INFO).

    Returns
    -------
    str
        The run_id used for this session.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())

    from config import LOGS_DIR, ensure_dirs
    ensure_dirs()

    log_file = LOGS_DIR / f"{run_id}.log"

    root = logging.getLogger()
    if root.handlers:
        return run_id  # Already configured; avoid duplicate handlers.

    root.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    # Rotating file handler -- max 50 MB, keep 5 backups
    fh = RotatingFileHandler(log_file, maxBytes=50 * 1024 * 1024, backupCount=5)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)

    return run_id
