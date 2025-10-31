import torch
import torch.nn as nn
from torchvision import models

class ScoresLayer(nn.Module):
    def __init__(self, input_dim, num_centers):
        super().__init__()
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.centers = nn.Parameter(torch.zeros(num_centers, input_dim), requires_grad=True)
        self.logsigmas = nn.Parameter(torch.zeros(num_centers), requires_grad=True)

    def forward(self, x):
        batch_size = x.size(0)
        out = x.view(batch_size, self.input_dim, 1, 1) # [batch, C, 1, 1]

        centers = self.centers[None, :, :, None, None]  # [1, K, C, 1, 1]
        diff = out.unsqueeze(1) - centers  # [batch, K, C, 1, 1]

        sum_diff = torch.sum(diff, dim=2)  # [batch, K, 1, 1]
        sign = torch.sign(sum_diff)

        squared_diff = torch.sum(diff ** 2, dim=2)  # [batch, K, 1, 1]

        logsigmas = nn.functional.relu(self.logsigmas)
        denominator = 2 * torch.exp(2 * logsigmas)
        part1 = (sign * squared_diff) / denominator.view(1, -1, 1, 1)

        part2 = self.input_dim * logsigmas
        part2 = part2.view(1, -1, 1, 1)

        scores = part1 + part2
        output = scores.sum(dim=(1, 2, 3)).view(-1, 1)  # [batch, 1]

        return output

class ImageClassifier(nn.Module):
    def __init__(self, settings):
        super().__init__()
        if settings.arch == 'baseline':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

        elif settings.arch == 'nodown':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            # Replace first conv layer to avoid downsampling
            new_conv = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
            new_conv.weight = nn.Parameter(self.backbone.conv1.weight)
            self.backbone.conv1 = new_conv
            # out_features = self.backbone.fc.in_features
            # self.backbone.fc = nn.Identity()
            # self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
            # print(self.backbone.fc.in_features)
            # quit()
            self.backbone.fc = nn.Sequential(nn.Linear(self.backbone.fc.in_features, 128), nn.Dropout(0.5))#, self.backbone.fc.in_features)
            # self.backbone.fc.weight.data.copy_(torch.eye(self.backbone.fc.in_features))
            # self.backbone.fc.bias.data.zero_()

        else:
            raise NotImplementedError('Model not recognized')
        
        if settings.freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

        self.prototype = settings.prototype

        if self.prototype:
            self.proto = ScoresLayer(input_dim=self.backbone.fc[0].out_features, num_centers=settings.num_centers)
            for param in self.proto.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)
        # if self.noise and label is not None:
        #     noise = torch.rand_like(x) > 0.5
        #     noise = (noise * label.unsqueeze(1).repeat(1, x.size(1))) == 1
        #     # if noise = 1 then x = -x only if label is 1
        #     x = torch.where(noise, -x, x)

        if self.prototype:
            x = self.proto(x)  
            
        # x = torch.where(x<2, nn.functional.reluw(x), nn.functional.sigmoid(x-2))

        return x
        