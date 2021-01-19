import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# import resnet as models


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        # change input channels of first layers to 1
        resnet_children = list(resnet.children())

        resnet_children[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # basic_blok.conv2 = nn.Conv2d(512, 784, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        # basic_blok.bn2 = nn.BatchNorm2d(784, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.features = nn.Sequential(*resnet_children[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x
