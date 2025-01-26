import torch
import torchvision.models as models
from torch import nn, optim
from art.estimators.classification import PyTorchClassifier

def load_model():
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

def create_classifier(model):
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(-1, 1),  # Adjust for normalized CIFAR-10
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.parameters(), lr=0.001),
        input_shape=(3, 32, 32),
        nb_classes=10
    )
    return classifier
