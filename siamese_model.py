import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class SiameseResNet18(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)

        # Remove the original ImageNet classification layer.
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        # Four 512-dimensional feature vectors.
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

    def forward(self, pre_image, post_image):
        # Shared backbone processes both images.
        pre_features = self.backbone(pre_image).flatten(1)
        post_features = self.backbone(post_image).flatten(1)

        # Compare the two feature representations.
        difference = torch.abs(pre_features - post_features)
        product = pre_features * post_features

        combined = torch.cat(
            [pre_features, post_features, difference, product],
            dim=1
        )

        return self.classifier(combined)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test architecture without downloading pretrained weights.
    model = SiameseResNet18(pretrained=False).to(device)
    model.eval()

    pre = torch.randn(2, 3, 128, 128).to(device)
    post = torch.randn(2, 3, 128, 128).to(device)

    with torch.no_grad():
        output = model(pre, post)

    print("Device:", device)
    print("Input shape:", pre.shape)
    print("Output shape:", output.shape)
    print("Expected output: [2, 4]")
    print("Model parameters:", sum(p.numel() for p in model.parameters()))

    assert output.shape == (2, 4)
    print("ARCHITECTURE TEST PASSED")
