import functools
import torch
import torch.nn as nn
from networks.resnet import resnet50
from networks.base_model import BaseModel, init_weights
import os


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if not hasattr(self, 'device'):
            self.device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')


        # if self.isTrain and not opt.continue_train:
        #     self.model = resnet50(pretrained=False, num_classes=1)

        # if not self.isTrain or opt.continue_train:
        #     self.model = resnet50(num_classes=1)
        self.model = resnet50(num_classes=1)

        if self.isTrain and getattr(opt, 'ft', False):
            # Bootstrap from pretrained model
            base_path = os.path.join('checkpoint', opt.name, 'weights', 'best.pt')
            if os.path.isfile(base_path):
                print(f"[FT] Loading base-trained weights from {base_path}")
                state = torch.load(base_path, map_location=opt.device)
                sd = state['model'] if isinstance(state, dict) and 'model' in state else state
                self.model.load_state_dict(sd, strict=True)
            else:
                print(f"[FT] WARNING: no checkpoint found at {base_path}. "
                      f"Training from random init instead.")


        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        # if not self.isTrain or opt.continue_train:
        #     self.load_networks(opt.epoch)
        # self.model.to(opt.gpu_ids[0])

        # check numbers of trainable parameters
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.total_params = sum(p.numel() for p in self.model.parameters())
        pct = 100 * self.trainable_params / self.total_params if self.total_params else 0.0
        print(f"Trainable parameters: {self.trainable_params:,} / {self.total_params:,} ({pct:.2f}%)")

        self.model.to(opt.device)
 

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            # param_group['lr'] *= 0.9
            param_group['lr'] /= 10 # same as r50nd
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/0.9} to {param_group["lr"]}')
        print('*'*25)
        return True
    
    # R50nd adjust learning rate
    # def adjust_learning_rate(self, min_lr=1e-6):
    #     for param_group in self.optimizer.param_groups:
    #         param_group["lr"] /= 10.0
    #         if param_group["lr"] < min_lr:
    #             return False
    #     return True

    def set_input(self, input):
        # self.input = input[0].to(self.device)
        # self.label = input[1].to(self.device).float()
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

