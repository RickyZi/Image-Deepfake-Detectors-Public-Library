'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
''' 

import os
import torch
import numpy as np
import tqdm
from networks import create_architecture, count_parameters

class TrainingModel(torch.nn.Module):

    def __init__(self, opt):
        super(TrainingModel, self).__init__()

        self.opt = opt
        self.total_steps = 0
        self.save_dir = (os.path.join('checkpoint', opt.name,'weights'))
        self.device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

        self.model = create_architecture(opt.arch, pretrained=True,  num_classes=1)
        num_parameters = count_parameters(self.model)
        print(f"Arch: {opt.arch} with #trainable {num_parameters}")

        self.loss_fn = torch.nn.BCEWithLogitsLoss().to(self.device)
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)

        self.model.to(self.device)

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] /= 10.0
            if param_group["lr"] < min_lr:
                return False
        return True

    def get_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def train_on_batch(self, data):
        self.total_steps += 1
        self.model.train()
        input = data['img'].to(self.device)
        label = data['target'].to(self.device).float()
        output = self.model(input)
        if len(output.shape) == 4:
            ss = output.shape
            loss = self.loss_fn(
                output,
                label[:, None, None, None].repeat(
                    (1, int(ss[1]), int(ss[2]), int(ss[3]))
                ),
            )
        else:
            loss = self.loss_fn(output.squeeze(1), label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.cpu()

    def save_networks(self, epoch):
        save_filename = f'{epoch}.pt'
        save_path = os.path.join(self.save_dir, save_filename)

        # serialize model and optimizer to dict
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
        }

        torch.save(state_dict, save_path)

    def predict(self, data_loader):
        model = self.model.eval()
        with torch.no_grad():
            y_true, y_pred, y_path = [], [], []
            for data in tqdm.tqdm(data_loader):
                img = data['img']
                label = data['target'].cpu().numpy()
                paths = list(data['path'])
                out_tens = model(img.to(self.device)).cpu().numpy()[:, -1]
                assert label.shape == out_tens.shape

                y_pred.extend(out_tens.tolist())
                y_true.extend(label.tolist())
                y_path.extend(paths)

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return y_true, y_pred, y_path
