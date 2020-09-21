import torch
import torch.nn as nn
import torchvision
from self_attention import AttentionModule


NUM_CLASSES=4
HIDDEN_SIZE = 256

class MusicSelfAttModel(nn.Module):
    def __init__(self):
        super(MusicSelfAttModel, self).__init__()
        self.mirex = MobileNetV2(4).cuda()
        self.att_model = nn.Sequential(
            AttentionModule()
            )

        
    def forward(self,x):
        numBands=96
        x = x.view(-1, numBands, 16, 256) # 16*256=4096 input

        x = x.permute(0,2,1,3)
        x = x.contiguous().view(-1,numBands,256)

        x = x.unsqueeze(1)
        x = self.mirex(x)
        att = x.view(-1, 16, 256)
        att = self.att_model(att)
        clf = x.view(-1,256)
        return att,clf



class MobileNetV2(nn.Module):    
    def __init__(self, num_classes):
        super().__init__()

        self.bw2col = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 10, 1, padding=0), nn.ReLU(),
            nn.Conv2d(10, 3, 1, padding=0), nn.ReLU())

        self.mv2 = torchvision.models.mobilenet_v2(pretrained=True)
        self.out_conv = nn.Sequential(
            nn.Conv2d(1280, 512, 3), nn.ReLU(),
            nn.Conv2d(512, 256, 1), nn.ReLU())

    def forward(self, x):
        x = self.bw2col(x)
        x = self.mv2.features(x)
        x = self.out_conv(x)
        x = x.max(dim=-1)[0].max(dim=-1)[0]
        return x

class ClassClassifierMobileNetV2(nn.Module):

    def __init__(self):
        super().__init__()

        # Define classifier network
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_CLASSES),
            nn.Sigmoid())

    def forward(self, input):
        clf = self.classifier(input)
        clf = clf.view(-1, 16, NUM_CLASSES)
        clf = clf.mean(dim=1)
        return clf

class ClassClassifierSelfAttention(nn.Module):

    def __init__(self):
        super().__init__()

        # Define classifier network
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_SIZE, NUM_CLASSES),
            nn.Sigmoid()
            )
    def forward(self, input):
        clf = self.classifier(input)
        return clf



class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer.
    Taken directly from:
    https://github.com/CuthbertCai/pytorch_DANN/blob/master/models/models.py
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

class DomainClassifierMobileNetV2(nn.Module):

    def __init__(self):
        super().__init__()

        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 2),
            nn.LogSoftmax()
        )
    def forward(self, input, constant):
        input = GradReverse.grad_reverse(input, constant)
        logits = self.domain_classifier(input)
        logits = logits.view(-1, 16, 2)
        logits = logits.mean(dim=1)
        return logits

class DomainClassifierSelfAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.domain_classifier = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 2),
            nn.LogSoftmax()
        )

    def forward(self, input, constant):
        input = GradReverse.grad_reverse(input, constant)
        logits = self.domain_classifier(input)
        return logits

