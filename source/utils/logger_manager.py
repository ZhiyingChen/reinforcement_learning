# -*- coding: utf-8 -*-
from __future__ import annotations
import logging
import os
from torch.utils.tensorboard import SummaryWriter

class LoggerManager:
    """
    统一管理 logging 与 tensorboard writer。
    """
    def __init__(self, log_dir: str = "logs/", use_tensorboard: bool = True):
        os.makedirs(log_dir, exist_ok=True)

        # ---- Python logging ----
        self.logger = logging.getLogger(f"RLLogger")
        self.logger.setLevel(logging.INFO)

        fmt = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p",
        )

        file_handler = logging.FileHandler(os.path.join(log_dir, "run.log"), mode="w", encoding="utf-8")
        file_handler.setFormatter(fmt)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # ---- Tensorboard Writer ----
        self.writer = SummaryWriter(log_dir) if use_tensorboard else None

    def log(self, msg: str):
        self.logger.info(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def add_scalar(self, tag: str, value: float, step: int):
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()