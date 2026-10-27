import os
import logging
import shutil
import torch.distributed as dist 

class RankFilter(logging.Filter): 
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        return dist.get_rank() == self.rank

def create_logger(log_path):
    if os.path.isdir(os.path.dirname(log_path)):
        log_path = check_if_log_file_exists(log_path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    else:
        os.makedirs(os.path.dirname(log_path))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(log_path)
    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    
    logger.addHandler(fh) 

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