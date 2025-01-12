import torch
import torch.nn as nn
from cnnrfvis import CatchEachOp


class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()

        # 第一层卷积
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )
        # 第二层卷积
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(2, 2),
            stride=(1, 1),
            padding=(1, 1),
        )
        # 第三层卷积
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
        )

        # 全连接层
        self.fc1 = nn.Linear(64 * 9 * 9, 128)  # 假设输入图像大小为 32x32
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        print("After conv1:", x.shape)
        x = torch.relu(self.conv2(x))
        print("After conv2:", x.shape)
        x = torch.relu(self.conv3(x))
        print("After conv3:", x.shape)

        x = x.view(x.size(0), -1)
        print("After flattening:", x.shape)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_simple_stride_cnn_rf_info():
    # 实例化模型
    model = CustomCNN(num_classes=10)
 
    # 随机生成输入张量
    input_tensor = torch.randn(1, 3, 32, 32)

    with CatchEachOp(verbose=False) as results, torch.no_grad():
        # 前向传播
        model(input_tensor)
    
    return results

if __name__ == "__main__":
    results = get_simple_stride_cnn_rf_info()
    print(type(results.rf_dict))
    print("Done!")
