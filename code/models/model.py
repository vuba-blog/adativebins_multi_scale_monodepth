from tkinter.tix import Select
from turtle import forward
import torch
import torch.nn as nn

from mmcv.runner import load_checkpoint
from .mit import mit_b4


class GLPDepth(nn.Module):
    def __init__(self, max_depth=10.0, is_train=False):
        super().__init__()
        self.max_depth = max_depth
        self.cnn_backbone = EfficientNet()
        self.encoder = mit_b4()
        if is_train:            
            ckpt_path = './code/models/weights/mit_b4.pth'
            try:
                load_checkpoint(self.encoder, ckpt_path, logger=None)
            except:
                import gdown
                print("Download pre-trained encoder weights...")
                id = '1BUtU42moYrOFbsMCE-LTTkUE-mrWnfG2'
                url = 'https://drive.google.com/uc?id=' + id
                output = './code/models/weights/mit_b4.pth'
                gdown.download(url, output, quiet=False)

        channels_in = [512, 320, 128]
        channels_out = 64
            
        self.decoder = Decoder(channels_in, channels_out)
    
        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        cnn_features = self.cnn_backbone(x)             
        conv1, conv2, conv3, conv4 = self.encoder(x)
        out = self.decoder(conv1, conv2, conv3, conv4, cnn_features)
        out_depth = self.last_layer_depth(out)
        out_depth = torch.sigmoid(out_depth) * self.max_depth

        return {'pred_d': out_depth}



class PixelWiseDotProduct(nn.Module):
    def __init__(self, values_dim=256, embedding_dim=256):
        super(PixelWiseDotProduct, self).__init__()
        self.values_dim = values_dim #change the dim of the values vector (encoder_1 = 64 to values_dim)
        self.embedding_dim = embedding_dim

        self.conv = nn.Conv2d(in_channels=embedding_dim, out_channels=values_dim, kernel_size=3,stride=1, padding=1)

        
    def forward(self, x, K):
        x = self.conv(x)
        # print("******* In PixelWiseProduct x shape: ", x.shape)
        # print("******* In PixelWiseProduct K shape: ", K.shape)
        n, c, h, w = x.size()
        _, cout, ck = K.size()
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))  # .shape = n, hw, cout
        return y.permute(0, 2, 1).view(n, cout, h, w)

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.cnn_feature_conv =nn.Conv2d(in_channels=2048, out_channels=out_channels, kernel_size=1)

        self.bot_conv = nn.Conv2d(
            in_channels=in_channels[0], out_channels=out_channels, kernel_size=1)
        
        self.skip_conv1 = nn.Conv2d(
            in_channels=in_channels[1], out_channels=out_channels, kernel_size=1)

        self.skip_conv2 = nn.Conv2d(
            in_channels=in_channels[2], out_channels=out_channels, kernel_size=1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.fusion0 = SelectiveFeatureFusion(out_channels)
        self.fusion1 = SelectiveFeatureFusion(out_channels)
        self.fusion2 = SelectiveFeatureFusion(out_channels)
        self.fusion3 = SelectiveFeatureFusion(out_channels)

    def forward(self, x_1, x_2, x_3, x_4, cnn_features):
        cnn_features_ = self.cnn_feature_conv(cnn_features)
        x_4_ = self.bot_conv(x_4)
        out = self.fusion0(cnn_features_, x_4_)
        out = self.up(out)

        x_3_ = self.skip_conv1(x_3)
        out = self.fusion1(x_3_, out)
        out = self.up(out)

        x_2_ = self.skip_conv2(x_2)

        out = self.fusion2(x_2_, out)
        out = self.up(out)

        out = self.fusion3(x_1, out)
        out = self.up(out)
        out = self.up(out)

        return out

class EfficientNet(nn.Module):
    """EfficientNet backbone.
    Following Adabins, this is a hack version of EfficientNet, where potential bugs exist.
    I recommend to utilize mmcls efficientnet presented at https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b0_8xb32_in1k.py
    Args:
        basemodel_name (str): Name of pre-trained EfficientNet. Default: tf_efficientnet_b5_ap.
        out_index List(int): Output from which stages. Default: [4, 5, 6, 8, 11].
    """
    def __init__(self, 
                 basemodel_name='tf_efficientnet_b5_ap',
                 out_index=[4, 5, 6, 8, 11]):
        super(EfficientNet, self).__init__()
        basemodel_name = 'tf_efficientnet_b5_ap'
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        self.original_model = basemodel
        self.out_index = out_index

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))

        out = []
        for index in self.out_index:
            out.append(features[index])

        #return the features at the stage 11 of EfficientNet as the CNN features.
        out = out[-1]
        return out


class SelectiveFeatureFusion(nn.Module):
    def __init__(self, in_channel=64):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channel*2),
                      out_channels=in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, 
                      out_channels=int(in_channel / 2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel / 2)),
            nn.ReLU())

        self.conv3 = nn.Conv2d(in_channels=int(in_channel / 2), 
                               out_channels=2, kernel_size=3, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_local, x_global):
        x = torch.cat((x_local, x_global), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        attn = self.sigmoid(x)

        out = x_local * attn[:, 0, :, :].unsqueeze(1) + \
              x_global * attn[:, 1, :, :].unsqueeze(1)

        return out

if __name__ == '__main__':
    model = GLPDepth(80, True)
    input = torch.rand(2, 3, 480, 640)
    out = model(input)
    # print("a")