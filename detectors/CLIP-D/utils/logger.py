# """
# utils/logger.py
# ───────────────
# Shared logger factory for CLIP-D training and testing.

# Creates a named Python logger that writes simultaneously to:
#   • stdout   — so you see output in the terminal / launcher log as before
#   • a .log file — persistent record of the run

# Usage
# ─────
#     from utils.logger import build_logger

#     logger = build_logger(
#         name     = "train",          # arbitrary; used to de-duplicate handlers
#         log_path = "logs/train_myrun.log",
#     )
#     logger.info("epoch=1  loss=0.4231  acc=0.7812  auc=0.8340  lr=1.0e-04")
#     logger.warning("LR dropped by 10×")
#     logger.error("something went wrong")

# Log-file location conventions (enforced in train.py / test.py):
#   Training : logs/train_<name>.log
#   Testing  : logs/test_<name>_<dataset>.log
# """

# import os
# import sys
# import logging


# def build_logger(name: str, log_path: str) -> logging.Logger:
#     """
#     Return a logger that tees every message to *both* stdout and a file.

#     Safe to call multiple times with the same `name` — existing handlers
#     are cleared first so you never get duplicate lines.

#     Args:
#         name     : logical name for the logger (e.g. "train", "test").
#                    Also used as the Python logger name so that
#                    logging.getLogger(name) returns the same instance.
#         log_path : absolute or relative path to the .log file.
#                    Parent directory is created automatically.

#     Returns:
#         logging.Logger instance with INFO level set by default.
#     """
#     os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

#     logger = logging.getLogger(name)

#     # Clear any handlers from a previous call (safe for re-imports /
#     # interactive sessions).
#     if logger.handlers:
#         logger.handlers.clear()

#     logger.setLevel(logging.DEBUG)

#     fmt = logging.Formatter(
#         fmt     = "%(asctime)s  %(levelname)-7s  %(message)s",
#         datefmt = "%Y-%m-%d %H:%M:%S",
#     )

#     # ── File handler ───────────────────────────────────────────────────
#     # mode='a' so that re-runs append rather than overwrite; useful when
#     # early stopping drops the LR and training resumes.
#     fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
#     fh.setLevel(logging.DEBUG)
#     fh.setFormatter(fmt)
#     logger.addHandler(fh)

#     # ── Console handler (stdout) ───────────────────────────────────────
#     # We attach to stdout (not stderr) so that tqdm progress bars, which
#     # write to stderr, don't interleave with our log lines in the terminal.
#     ch = logging.StreamHandler(sys.stdout)
#     ch.setLevel(logging.DEBUG)
#     ch.setFormatter(fmt)
#     logger.addHandler(ch)

#     # Don't propagate to the root logger — avoids double-printing when
#     # other libraries (open_clip, torch) also use the root logger.
#     logger.propagate = False

#     logger.info(f"Logger '{name}' writing to: {os.path.abspath(log_path)}")
#     return logger


# --------------------------------------------------------------------------- #
# taken from DFB github repo -> https://github.com/SCLBD/DeepfakeBench
import os
import logging
import shutil

import torch.distributed as dist # torch.distributed is a package that enables multi-process communication

class RankFilter(logging.Filter): # logging.Filter is a class that allows you to filter log records based on certain criteria (e.g. log level, message, etc.)
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        return dist.get_rank() == self.rank

def create_logger(log_path):
    # Create log directory if it does not exist
    if os.path.isdir(os.path.dirname(log_path)):
        log_path = check_if_log_file_exists(log_path)
        # print(log_path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    else:
        # If the directory does not exist, create it
        os.makedirs(os.path.dirname(log_path))

    # Create logger object
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Create file handler and set the formatter
    fh = logging.FileHandler(log_path)
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # asctime: human-readable time when the LogRecord was created
    # levelname: the log level of the LogRecord (e.g. INFO, WARNING, etc.)
    # message: the log message
    fh.setFormatter(formatter)

    # Add the file handler to the logger (write to file)
    logger.addHandler(fh) 
    # Ensure no stream handler is added to the logger (do not print to console)
    # for handler in logger.handlers:
    #     if isinstance(handler, logging.StreamHandler):
    #         logger.removeHandler(handler)

    return logger

def check_if_log_file_exists(log_path):
    # Check if the log file already exists and modify the log path if necessary
        base, ext = os.path.splitext(log_path)
        # counter = 1
        # print("base: ", base)
        # print("ext: ", ext)
        if 'test_output_v' not in base:
            counter = 1
        else:
            counter = int(base.split('_v')[-1]) + 1
        # print("counter: ", counter)
        while os.path.isfile(log_path):
            if counter == 1:
                log_path = f"{base}_v{counter}{ext}"
            else:
                log_path = log_path.replace(f'_v{counter-1}', f'_v{counter}')
            counter += 1
        return log_path