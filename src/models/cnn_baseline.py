import torch
import torch.nn as nn
from torchvision import models


class PneumoniaResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaResNet, self).__init__()
        # Load Pre-trained ResNet-50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Modify the final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# PyCharm Tip: Type 'main' and hit Tab to generate this block
if __name__ == "__main__":
    model = PneumoniaResNet()
    print(model)