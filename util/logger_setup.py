"""Unified logging setup for development / testing / production environments.

Usage: call setup_logging() once at application startup, before any logger usage.
"""

import logging
import logging.handlers
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Literal

_RESET = "\033[0m"
_COLORS: dict[int, str] = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[41m\033[37m",
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_EnvName = Literal["development", "testing", "production"]

_ENV_PRESETS: dict[str, dict] = {
    "development": {
        "console": True,
        "console_level": logging.DEBUG,
        "console_colored": True,
        "console_format": "%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d] - %(message)s",
        "file_level": logging.DEBUG,
        "file_name": "dev.log",
        "rotation_mb": 5,
        "retention_days": 3,
        "compress": False,
        "diagnose": True,
    },
    "testing": {
        "console": False,
        "console_level": logging.DEBUG,
        "console_colored": False,
        "console_format": "%(asctime)s %(levelname)s %(name)s - %(message)s",
        "file_level": logging.DEBUG,
        "file_name": "test.log",
        "rotation_mb": 10,
        "retention_days": 1,
        "compress": False,
        "diagnose": False,
    },
    "production": {
        "console": True,
        "console_level": logging.WARNING,
        "console_colored": False,
        "console_format": "%(asctime)s %(levelname)s %(name)s - %(message)s",
        "file_level": logging.INFO,
        "file_name": "app.log",
        "rotation_mb": 10,
        "retention_days": 7,
        "compress": True,
        "diagnose": False,
    },
}

_VALID_ENVS = tuple(_ENV_PRESETS)


def _get_log_dir() -> Path:
    log_dir = _PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


class _CleanupRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler with time-based cleanup and optional zip compression."""

    def __init__(
        self,
        filename: str,
        maxBytes: int = 0,
        backupCount: int = 5,
        retention_days: int = 7,
        compress: bool = False,
        **kwargs,
    ):
        super().__init__(filename, maxBytes=maxBytes, backupCount=backupCount, **kwargs)
        self.retention_days = retention_days
        self.compress = compress

    def doRollover(self) -> None:
        super().doRollover()
        self._cleanup_old()

    def _cleanup_old(self) -> None:
        base = Path(self.baseFilename)
        prefix = base.name
        cutoff = time.time() - self.retention_days * 86400

        for f in base.parent.iterdir():
            if not f.is_file() or f.name == prefix or not f.name.startswith(prefix):
                continue

            try:
                mtime = f.stat().st_mtime
                if mtime < cutoff:
                    f.unlink()
                    continue
            except FileNotFoundError:
                continue

            if self.compress and f.suffix != ".zip":
                self._compress_and_remove(f)

    @staticmethod
    def _compress_and_remove(log_file: Path) -> None:
        zip_path = log_file.with_suffix(log_file.suffix + ".zip")
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(log_file, log_file.name)
            os.utime(zip_path, (log_file.stat().st_mtime, log_file.stat().st_mtime))
            log_file.unlink()
        except Exception:
            pass


class _ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelno, "")
        if color:
            record.levelname = f"{color}{record.levelname}{_RESET}"
        return super().format(record)


_UVICORN_LOGGERS = ["uvicorn", "uvicorn.error", "uvicorn.access"]


def _hijack_uvicorn(handlers: list[logging.Handler]) -> None:
    for name in _UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        for h in list(uvicorn_logger.handlers):
            h.close()
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = False
        for h in handlers:
            uvicorn_logger.addHandler(h)
        uvicorn_logger.setLevel(logging.DEBUG)


def _install_diagnose_hook() -> None:
    original_hook = sys.excepthook

    def diagnose_hook(exc_type, exc_value, exc_tb):
        sys.stderr.write("\n── Diagnose: local variables ──\n")
        tb = exc_tb
        while tb:
            frame = tb.tb_frame
            sys.stderr.write(
                f'  File "{frame.f_code.co_filename}", line {tb.tb_lineno}, in {frame.f_code.co_name}\n'
            )
            for name, value in frame.f_locals.items():
                try:
                    sys.stderr.write(f"    {name} = {repr(value)[:200]}\n")
                except Exception:
                    sys.stderr.write(f"    {name} = <repr error>\n")
            tb = tb.tb_next
        sys.stderr.write("── End diagnose ──\n\n")
        original_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = diagnose_hook


def setup_logging(app_env: _EnvName | None = None) -> None:
    """Configure logging for the entire application.

    Must be called once at startup, before any logger usage.
    Reads APP_ENV from environment (default: 'development').
    """
    if app_env is None:
        app_env = os.getenv("APP_ENV", "development")
    if app_env not in _ENV_PRESETS:
        raise ValueError(
            f"Unknown APP_ENV={app_env!r}, expected one of {_VALID_ENVS}"
        )

    config = dict(_ENV_PRESETS[app_env])

    if app_env == "production":
        config["rotation_mb"] = int(os.getenv("LOG_ROTATION_MB", str(config["rotation_mb"])))
        config["retention_days"] = int(os.getenv("LOG_RETENTION_DAYS", str(config["retention_days"])))

    log_dir = _get_log_dir()

    root = logging.getLogger()
    for h in list(root.handlers):
        h.close()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    handlers: list[logging.Handler] = []

    if config["console"]:
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(config["console_level"])
        fmt_cls = _ColoredFormatter if config.get("console_colored") else logging.Formatter
        console.setFormatter(fmt_cls(config["console_format"]))
        root.addHandler(console)
        handlers.append(console)

    file_path = str(log_dir / config["file_name"])
    rotation_bytes = config["rotation_mb"] * 1024 * 1024
    file_handler = _CleanupRotatingFileHandler(
        file_path,
        maxBytes=rotation_bytes,
        backupCount=20,
        retention_days=config["retention_days"],
        compress=config["compress"],
    )
    file_handler.setLevel(config["file_level"])
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d] - %(message)s"
        )
    )
    root.addHandler(file_handler)
    handlers.append(file_handler)

    _hijack_uvicorn(handlers)

    if config["diagnose"]:
        _install_diagnose_hook()

    logging.getLogger(__name__).info(
        "Logging configured: APP_ENV=%s rotation=%dMB retention=%dd compress=%s",
        app_env,
        config["rotation_mb"],
        config["retention_days"],
        config["compress"],
    )
