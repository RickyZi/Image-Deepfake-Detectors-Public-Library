import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger
from tqdm import tqdm
import random
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True

    return val_opt


if __name__ == '__main__':
    opt_train = TrainOptions().parse()
    seed_torch(100)

    print('  '.join(list(sys.argv)) )
    opt_val = get_val_opt()

    train_loader = create_dataloader(opt_train, split='train')
    val_loader = create_dataloader(opt_val, split='val')

    
    model = Trainer(opt_train)
    
    model.train()
    print(f'cwd: {os.getcwd()}')
    for epoch in range(opt_train.niter):
        if epoch > 0:
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0

            #for i, data in enumerate(train_loader):
            with tqdm(train_loader, unit='batch', mininterval=0.5) as tepoch:
                tepoch.set_description(f'Epoch {epoch}', refresh=False)
                for i, data in enumerate(tepoch):
                    model.total_steps += 1
                    epoch_iter += opt_train.batch_size

                    model.set_input(data)
                    model.optimize_parameters()
                    tepoch.set_postfix(loss=model.loss.item())
                
            if epoch % opt_train.delr_freq == 0 and epoch != 0:
                print('changing lr at the end of epoch %d, iters %d' % (epoch, model.total_steps))
                model.adjust_learning_rate()
            

        # Validation
        model.eval()
        acc, ap = validate(model.model, val_loader)[:2]
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
        model.train()
        if epoch == 0:
            model.save_networks('best')
        elif acc >= model.best_acc:
            model.save_networks('best')
    
