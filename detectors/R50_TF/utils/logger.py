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