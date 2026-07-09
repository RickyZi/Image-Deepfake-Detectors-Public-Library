import sys
import os
import torch
import logging


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unnormalize(tens, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # assume tensor of shape NxCxHxW
    return tens * torch.Tensor(std)[None, :, None, None] + torch.Tensor(
        mean)[None, :, None, None]




class Logger(object):
    """Log stdout messages."""

    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "a")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        
        
def printSet(set_str):
    set_str = str(set_str)
    num = len(set_str)
    print("="*num*3)
    print(" "*num + set_str)
    print("="*num*3)

def check_if_log_file_exists(log_path):
    # Auto-version the log path if it already exists, so re-running the
    # same experiment doesn't append to (or overwrite) a previous log.
    base, ext = os.path.splitext(log_path)
    if '_v' not in base:
        counter = 1
    else:
        counter = int(base.split('_v')[-1]) + 1
    while os.path.isfile(log_path):
        if counter == 1:
            log_path = f"{base}_v{counter}{ext}"
        else:
            log_path = log_path.replace(f'_v{counter-1}', f'_v{counter}')
        counter += 1
    return log_path
 
 
def create_logger(log_path):
    if os.path.isdir(os.path.dirname(log_path)):
        log_path = check_if_log_file_exists(log_path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    else:
        os.makedirs(os.path.dirname(log_path))
 
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
 
    return logger


# add EarlyStopping implementation
class EarlyStopping:
    """Validation-triggered early stopping. Call with the current
    validation score each epoch; .early_stop becomes True once `patience`
    consecutive epochs pass without an improvement greater than `delta`."""
    def __init__(self, init_score=None, patience=1, verbose=False, delta=0, logger=None):
        self.best_score = init_score
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.logger = logger
        self.count_down = self.patience
        self.early_stop = False
 
    def __call__(self, score):
        if self.best_score is None:
            if self.verbose:
                print(f'Score set to {score:.6f}.')
            self.best_score = score
            self.count_down = self.patience
            return True
        elif score <= self.best_score + self.delta:
            self.count_down -= 1
            if self.verbose:
                print(f'EarlyStopping count_down: {self.count_down} on {self.patience}')
                if self.logger:
                    self.logger.info(f'EarlyStopping count_down: {self.count_down} on {self.patience}')
            if self.count_down <= 0:
                self.early_stop = True
            return False
        else:
            if self.verbose:
                print(f'Score increased from ({self.best_score:.6f} to {score:.6f}).')
                if self.logger:
                    self.logger.info(f'EarlyStopping count_down: {self.count_down} on {self.patience}')
            self.best_score = score
            self.count_down = self.patience
            return True
 
    def reset_counter(self):
        self.count_down = self.patience
        self.early_stop = False
