from torchvision.models import resnet50,ResNet50_Weights
import torch.nn as nn
import torch

class MyResnet50(nn.Module):
    def __init__(self, num_items=2,num_colors=4):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        del self.model.fc

        for name,param in self.model.named_parameters():
            if "layer4" not in name:
                param.requires_grad = False

        self.model.fc_item = nn.Linear(in_features=2048, out_features=num_items)
        self.model.fc_color = nn.Linear(in_features=2048, out_features=num_colors)
        # for name,param in self.model.named_parameters():
        #     print(name,param.requires_grad)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x1 = self.model.fc_item(x)
        x2 = self.model.fc_color(x)

        return x1,x2

    def forward(self, x):
        return self._forward_impl(x)




if __name__ == '__main__':

    model = MyResnet50(2,4)
    # input = torch.rand(1,3,224,224)
    # while True:
    #     item,color = model(input)
    #
    #     print(item)



