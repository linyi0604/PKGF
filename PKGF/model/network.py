import torch
import torch.nn as nn
from torchvision import models
import segmentation_models_pytorch as smp


class VGG16_PK(nn.Module):
    def __init__(self, input_channel, class_num, pretrained=True):
        super(VGG16_PK, self).__init__()
        
        self.pre_process = nn.Sequential(
            nn.Conv2d(input_channel, 3, (1, 1)),
            nn.ReLU(),
        )

        vgg16 = models.vgg16(pretrained=pretrained)
        self.feature = vgg16.features
        self.pool = vgg16.avgpool
        self.classifier = vgg16.classifier
        self.classifier[6] = nn.Linear(4096, class_num)

    def forward(self, x):
        x = self.pre_process(x)
        x = self.feature(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Unetpp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
    def forward(self, x):
        return self.model(x)



class Net(nn.Module):
    def __init__(self, input_channel, class_num, pretrained):
        super(Net, self).__init__()
        self.attention = Unetpp()
        self.grading = VGG16_PK(input_channel, class_num, pretrained=pretrained)

    def forward(self, x):
        attention = self.attention(x)
        grading = self.grading(torch.cat([attention, x], dim=1))

        return attention, grading
