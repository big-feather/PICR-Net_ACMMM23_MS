"""
@Project: PICR_Net
@File: modules/VGG.py
@Author: chen zhang
@Institution: Beijing JiaoTong University
"""
# import torch.nn as nn
import torchvision
# import torch
import mindspore
import mindspore.nn as nn


class VGG16(nn.Cell):
    def __init__(self):
        super(VGG16, self).__init__()
        conv1 = nn.SequentialCell()
        conv1.append(nn.Conv2d(3, 64, 3, 1,'pad', 1))
        conv1.append( nn.ReLU())
        conv1.append(nn.Conv2d(64, 64, 3, 1, 'pad',1))
        conv1.append( nn.ReLU())
        self.conv1 = conv1
        
        conv2 = nn.SequentialCell()
        conv2.append( nn.AvgPool2d(2, stride=2))
        conv2.append( nn.Conv2d(64, 128, 3, 1,'pad', 1))
        conv2.append( nn.ReLU())
        conv2.append( nn.Conv2d(128, 128, 3, 1, 'pad',1))
        conv2.append(nn.ReLU())
        self.conv2 = conv2

        conv3 = nn.SequentialCell()
        conv3.append( nn.AvgPool2d(2, stride=2))
        conv3.append( nn.Conv2d(128, 256, 3, 1, 'pad',1))
        conv3.append(nn.ReLU())
        conv3.append( nn.Conv2d(256, 256, 3, 1,'pad', 1))
        conv3.append( nn.ReLU())
        conv3.append( nn.Conv2d(256, 256, 3, 1, 'pad',1))
        conv3.append(nn.ReLU())
        self.conv3 = conv3

        conv4 = nn.SequentialCell()
        conv4.append( nn.AvgPool2d(2, stride=2))
        conv4.append( nn.Conv2d(256, 512, 3, 1,'pad', 1))
        conv4.append( nn.ReLU())
        conv4.append( nn.Conv2d(512, 512, 3, 1,'pad', 1))
        conv4.append( nn.ReLU())
        conv4.append( nn.Conv2d(512, 512, 3, 1,'pad', 1))
        conv4.append( nn.ReLU())
        self.conv4 = conv4

        conv5 = nn.SequentialCell()
        conv5.append( nn.AvgPool2d(2, stride=2))
        conv5.append( nn.Conv2d(512, 512, 3, 1,'pad', 1))
        conv5.append( nn.ReLU())
        conv5.append( nn.Conv2d(512, 512, 3, 1,'pad', 1))
        conv5.append( nn.ReLU())
        conv5.append( nn.Conv2d(512, 512, 3, 1,'pad', 1))
        conv5.append( nn.ReLU())
        self.conv5 = conv5

        # vgg_16 = torchvision.models.vgg16(pretrained=True)
        # self._initialize_weights(vgg_16)

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def _initialize_weights(self, vgg_16):
        features = [
            self.conv1.conv1_1, self.conv1.relu1_1,
            self.conv1.conv1_2, self.conv1.relu1_2,
            self.conv2.pool1,
            self.conv2.conv2_1, self.conv2.relu2_1,
            self.conv2.conv2_2, self.conv2.relu2_2,
            self.conv3.pool2,
            self.conv3.conv3_1, self.conv3.relu3_1,
            self.conv3.conv3_2, self.conv3.relu3_2,
            self.conv3.conv3_3, self.conv3.relu3_3,
            self.conv4.pool3,
            self.conv4.conv4_1, self.conv4.relu4_1,
            self.conv4.conv4_2, self.conv4.relu4_2,
            self.conv4.conv4_3, self.conv4.relu4_3,
            self.conv5.pool4,
            self.conv5.conv5_1, self.conv5.relu5_1,
            self.conv5.conv5_2, self.conv5.relu5_2,
            self.conv5.conv5_3, self.conv5.relu5_3,
        ]
        for l1, l2 in zip(vgg_16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data


def VGG_2(pretrained=True):
    import mindcv

    net = mindcv.create_model('vgg16', pretrained=pretrained)
    # if pretrained:
    #     print("The backbone model loads the pretrained parameters...")
    #     model_dict = net.parameters_and_names()
    #     state_dict = {}
    #     for name, param in model_dict:
    #         state_dict[name] = param
    #     pretrained_dict = mindspore.load_checkpoint("./pretrain/vgg16-95697531.ckpt")
    #     # 1. filter out unnecessary keys
    #     filter_dict = {}
    #     for key, value in pretrained_dict.copy().items():
    #         if key in state_dict:
    #             filter_dict[key] = value
    #     # pretrained_dict = {k: v for k, v in pretrained_dict.copy().items() if k in state_dict}
    #     print("VGG The number of loaded parameters: ", len(filter_dict))
    #     # 2. overwrite entries in the existing state dict
    #     state_dict.update(filter_dict)

        # 3. load the new state dict
        # mindspore.train.serialization.load_param_into_net(net, state_dict)

    low_feature_extract_224 = nn.SequentialCell(net.features[:4])
    low_feature_extract_112 = nn.SequentialCell(net.features[4:9])

    return low_feature_extract_224, low_feature_extract_112


class LowFeatureExtract(nn.Cell):
    def __init__(self):
        super(LowFeatureExtract, self).__init__()
        (
            self.low_feature_extract_224,
            self.low_feature_extract_112
        ) = VGG_2(pretrained=True)

    def construct(self, x):
        feature_224 = self.low_feature_extract_224(x)
        feature_112 = self.low_feature_extract_112(feature_224)

        return feature_224, feature_112


if __name__ == '__main__':
    x = mindspore.ops.randn(1, 3, 224, 224)
    backbone = LowFeatureExtract()
    out = backbone(x)
    print(out[0].shape)