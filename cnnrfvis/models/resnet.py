import torch
from torchvision.models import resnet18
from cnnrfvis import CatchEachOp

model = resnet18(pretrained=True)
model.eval()

batch_size = 1
height, width = 224, 224
input_tensor = torch.randn(batch_size, 3, height, width)

with CatchEachOp():
    print(model)
    model(input_tensor)
