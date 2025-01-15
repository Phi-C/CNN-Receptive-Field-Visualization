import torch
from torchvision.models import resnet18
from cnnrfvis import CatchEachOp


def get_resnet18_rf_info():
    model = resnet18(pretrained=True)
    model.eval()

    batch_size = 1
    height, width = 224, 224
    input_tensor = torch.randn(batch_size, 3, height, width)

    with CatchEachOp(verbose=False) as results, torch.no_grad():
        # print(model)
        model(input_tensor)

    return results


if __name__ == "__main__":
    results = get_resnet18_rf_info()
    print("Done!")
