import torch
import torch.nn as nn
from torch.utils.data import Dataset

# =============================================================================
# Helper Modules (Utility Layers)
# =============================================================================

class ApsPool(nn.Module):
    """
    Adaptive Polyphase Sampling or Anti-aliasing Pool placeholder.
    Downsamples the feature map by stride 2.
    """

    def __init__(self):
        super(ApsPool, self).__init__()
        self.stride = 2

    def forward(self, x):
        if x.size(2) <= 5:
            return x
        norm_ind = 2
        if x.size(2) % 2 != 0:
            final = x.size(2) - 1
        else:
            final = x.size(2)
        x_0 = x[:, :, ::self.stride, :]
        x_1 = x[:, :, 1::self.stride, :]
        # x_0 = x[:, :, ::2, ::2]
        # x_1 = x[:, :, 1::2, ::2]
        # x_2 = x[:, :, ::2, 1::2]
        # x_3 = x[:, :, 1::2, 1::2]
        xpoly_0 = x_0.reshape(x.shape[0], -1)
        xpoly_1 = x_1.reshape(x.shape[0], -1)
        # xpoly_2 = x_2.reshape(x.shape[0], -1)
        # xpoly_3 = x_3.reshape(x.shape[0], -1)
        norm0 = torch.norm(xpoly_0, dim=1, p=norm_ind)
        norm1 = torch.norm(xpoly_1, dim=1, p=norm_ind)
        # norm2 = torch.norm(xpoly_2, dim=1, p=norm_ind)
        # norm3 = torch.norm(xpoly_3, dim=1, p=norm_ind)
        all_norms = torch.stack([norm0, norm1], dim=1)
        max_ind = torch.argmax(all_norms, dim=1)
        xpoly_combined = torch.stack([x_0, x_1], dim=4)
        B = xpoly_combined.shape[0]
        output = xpoly_combined[torch.arange(B), :, :, :, max_ind]
        return output


class ResidualBlock(nn.Module):
    """
    Standard ResBlock to demonstrate deep feature extraction capabilities.
    """

    def __init__(self, in_channels, out_channels, stride=1, groups=32):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        )
        if in_channels != out_channels:
            self.conv1.add_module("downsample", ApsPool())
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                ApsPool(),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out1 = self.relu(self.bn1(self.conv1(x)))
        out1 = self.bn2(self.conv2(out1))
        out1 += self.shortcut(x)
        out1 = self.relu(out1)
        return out1


# =============================================================================
# Core Network Architecture
# =============================================================================

class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model_cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 7, (1, 2), padding=3),
            ApsPool(),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, (1, 2), padding=1),
            ApsPool(),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            ResidualBlock(32, 64, (1, 2), 16),
            ResidualBlock(64, 64, 1, 16),
            ResidualBlock(64, 128, (1, 2)),
            ResidualBlock(128, 128, 1),
            ResidualBlock(128, 256, (1, 2)),
            ResidualBlock(256, 256, 1),
            torch.nn.AdaptiveAvgPool2d(1)  # 256X1
        )
        self.model_lstm = torch.nn.LSTM(input_size=256, hidden_size=512, num_layers=2, batch_first=True)
        self.model_mlp = torch.nn.Sequential()
        self.model_mlp.add_module("linear1", torch.nn.Linear(512, 4 * 512))
        self.model_mlp.add_module("relu1", torch.nn.ReLU())
        self.model_mlp.add_module("linear2", torch.nn.Linear(4 * 512, 55))
        self.model_mlp.add_module("softmax", torch.nn.LogSoftmax(dim=1))

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        img_h = x.size(3)
        img_w = x.size(4)
        x = x.view(-1, 1, img_h, img_w)
        x = self.model_cnn(x)
        x = x.view(batch_size, seq_len, -1)
        output, (h, c) = self.model_lstm(x)
        x = self.model_mlp(h[1])
        return x

class MyDataSet(Dataset):
    def __init__(self, data1, label1):
        self.data = data1
        self.label = label1

    def __getitem__(self, index):
        nmb = index
        label2 = self.label[nmb]
        data2 = self.data[nmb]
        return torch.tensor(data2), torch.tensor(label2), index

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    # Sanity Check to print model structure
    model = Net()
    print(model)

