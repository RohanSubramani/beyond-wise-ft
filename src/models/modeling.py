import torch
import copy
import time

# import clip.clip as clip
import open_clip.src.open_clip as open_clip

from src.models import utils


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        try:
            self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
                args.model, pretrained='laion400m_e32',data_augmentation_number=args.data_augmentation)
            print("Using pretrained=laion400m_e32")
        except RuntimeError:
            try:
                self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
                args.model,pretrained='laion400m_e31',data_augmentation_number=args.data_augmentation)
                print("Using pretrained=laion400m_e31")
            except RuntimeError:
                print("Pretrained model not found!")
                cont = input("Would you like to continue anyway? 1 = yes, 2 = no\n")
                if int(cont) == 2:
                    print("Stopping run.")
                    raise RuntimeError("No pretrained model found.")
                self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
                    args.model,data_augmentation_number=args.data_augmentation)  # No pretrained sometimes? Expensive to train from scratch though.
        self.model.to(args.device)
                                                                 # clip.load(args.model, args.device, jit=False)
        
        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs):
        try:
            if self.process_images:
                inputs = self.image_encoder(inputs)
            outputs = self.classification_head(inputs)
        except RuntimeError:
            self.process_images=True
            if self.process_images:
                inputs = self.image_encoder(inputs)
            outputs = self.classification_head(inputs)
        return outputs

    def save(self, filename):
        # print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)

import torch.nn as nn
import torch.nn.functional as F

class ImageClassifier2(torch.nn.Module):  # Forward function maps to a list of alphas
    def __init__(self, image_encoder, classification_head, process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, image, all_logits):
        if self.process_images:
            image = self.image_encoder(image)
        out = self.classification_head(image,all_logits)
        return out

    def save(self, filename):
        # print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)

class ClassificationHead2(torch.nn.Linear):
    def __init__(self, normalize, weights, output_size, biases=None):
        _, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs, all_logits):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True) 
        out = super().forward(inputs)
        out = nn.Softmax(dim=0)(out)
        out = out @ all_logits  # Weighted average of logits from the models being ensembled
        # out = nn.Softmax(dim=0)(out)  # Loss function is applied to logits
        return out

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return utils.torch_load(filename)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class StackingResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=3, num_classes=2):
        super(StackingResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion,num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, all_logits):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = nn.Softmax(dim=0)(out)
        out = out @ all_logits
        # out = nn.Softmax(dim=0)(out)  # Loss function is applied to logits
        return out


def ResNet18(in_channels=3, num_classes=10):
    return StackingResNet(BasicBlock, [2, 2, 2, 2], in_channels, num_classes)


def ResNet34(in_channels=3, num_classes=10):
    return StackingResNet(BasicBlock, [3, 4, 6, 3], in_channels, num_classes)


def ResNet50(in_channels=3, num_classes=10):
    return StackingResNet(Bottleneck, [3, 4, 6, 3], in_channels, num_classes)


def ResNet101(in_channels=3, num_classes=10):
    return StackingResNet(Bottleneck, [3, 4, 23, 3], in_channels, num_classes)

def ResNet152(in_channels=3, num_classes=10):
    return StackingResNet(Bottleneck, [3, 8, 36, 3], in_channels, num_classes)