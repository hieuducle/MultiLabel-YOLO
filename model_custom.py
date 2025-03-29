import torch
import torch.nn as nn
import cv2
import numpy as np
class SimpleCNN(nn.Module):

    def __init__(self, num_items,num_colors):
        super().__init__()
        self.conv1 = self.make_block(in_channels=3, out_channels=8)
        self.conv2 = self.make_block(in_channels=8, out_channels=16)
        self.conv3 = self.make_block(in_channels=16, out_channels=32)
        self.conv4 = self.make_block(in_channels=32, out_channels=64)
        self.conv5 = self.make_block(in_channels=64, out_channels=128)

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=6272, out_features=512),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU()
        )
        self.fc_item = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=num_items),
        )
        self.fc_color = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=num_colors),
        )


    def make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,kernel_size=3,stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x1 = self.fc_item(x)
        x2 = self.fc_color(x)
        return x1,x2




if __name__ == '__main__':
    input_tensor = torch.rand((1, 3, 224, 224))
    # img = cv2.imread("/home/amin/PycharmProjects/PythonProject/test.jpg")
    # img = cv2.resize(img, (32, 32))
    #
    # img = np.transpose(img, (2, 0, 1))
    # img = torch.from_numpy(img)

    model = SimpleCNN(3,5)
    item,color = model(input_tensor)
    print(color.shape)
    # print("goc {}".format(output))
    # soft_max = nn.Softmax(dim=1)
    # out = soft_max(output)
    # max_idx_out = torch.argmax(output, dim=1)
    # max_idx_soft = torch.argmax(out, dim=1)
    # print("soft {}".format(max_idx_soft.item()))
    # print("argmax {}".format(max_idx_out))
