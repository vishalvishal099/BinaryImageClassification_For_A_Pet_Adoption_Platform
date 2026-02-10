"""
CNN Model Architectures for Cats vs Dogs Classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SimpleCNN(nn.Module):
    """
    Simple CNN baseline model for binary image classification.
    
    Architecture:
    - 3 convolutional blocks with batch normalization and max pooling
    - 2 fully connected layers
    - Dropout for regularization
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        dropout_rate: float = 0.5
    ):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 224 -> 112
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112 -> 56
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 56 -> 28
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28 -> 14
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class ImprovedCNN(nn.Module):
    """
    Improved CNN with residual connections for better performance.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        dropout_rate: float = 0.5
    ):
        super(ImprovedCNN, self).__init__()
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 224 -> 56
        )
        
        # Residual blocks
        self.layer1 = self._make_residual_block(64, 64)
        self.layer2 = self._make_residual_block(64, 128, stride=2)  # 56 -> 28
        self.layer3 = self._make_residual_block(128, 256, stride=2)  # 28 -> 14
        self.layer4 = self._make_residual_block(256, 512, stride=2)  # 14 -> 7
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights()
    
    def _make_residual_block(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1
    ) -> nn.Module:
        """Create a residual block."""
        layers = []
        
        # First conv
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Second conv
        layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.stem(x)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        x = F.relu(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


def get_model(
    model_name: str = "simple_cnn",
    num_classes: int = 2,
    dropout_rate: float = 0.5,
    pretrained: bool = False
) -> nn.Module:
    """
    Get a model by name.
    
    Args:
        model_name: Name of the model ('simple_cnn', 'improved_cnn', 'resnet18')
        num_classes: Number of output classes
        dropout_rate: Dropout rate
        pretrained: Whether to use pretrained weights (for transfer learning)
        
    Returns:
        PyTorch model
    """
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_name == "improved_cnn":
        return ImprovedCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_name == "resnet18":
        from torchvision import models
        model = models.resnet18(pretrained=pretrained)
        # Replace final fully connected layer
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.fc.in_features, num_classes)
        )
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
