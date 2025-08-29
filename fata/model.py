import torch
import torch.nn as nn
import torchvision.models as models

def setup_model(model_name: str = "resnet50"):
    """
    Loads a pre-trained model, sets up hooks, and configures adaptable parameters.
    """
    if model_name != "resnet50":
        raise NotImplementedError("Currently only supports resnet50")

    # 1. Load pre-trained model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    # 2. Split the model for the dual-branch forward pass
    model_part1 = nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool,
        model.layer1, model.layer2, model.layer3
    )
    model_part2 = nn.Sequential(model.layer4, model.avgpool, nn.Flatten(), model.fc)

    # 3. Configure which parameters to adapt (only BatchNorm affine params)
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(True)
            if module.weight is not None:
                module.weight.requires_grad = True
            if module.bias is not None:
                module.bias.requires_grad = True

    # 4. Setup the optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.00025 # Lower LR for single-image adaptation
    )

    print("ResNet-50 model prepared for Test-Time Adaptation.")
    return model, model_part1, model_part2, optimizer

