from turtle import forward
import torch
import torch.nn as nn

from mmcv.runner import load_checkpoint
from mit import mit_b4

class GLPDepth(nn.Module):
    def __init__(self, max_depth=10.0, is_train=False):
        super().__init__()
        self.max_depth = max_depth

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

        # self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)

        self.ada_bins = adaptive_bins(in_channels=100, out_channels=100, embedding_dim=512, dim_out=256)

    
        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):      
        print("****** Model input shape: ", x.shape)  
        print("****** lenth of Encoder output: ", len(self.encoder(x)))  
        convs, embed_sequence = self.encoder(x)

        #adabins section
        b_ = self.ada_bins(embed_sequence)
        print("****** Shape of b: ", b_.shape)
        #######


        print("****** Shape of embed_sequence: ", embed_sequence.shape) 
        conv1, conv2, conv3, conv4 = convs
        print("****** Output from the Encoder (mixtransformer): ", conv1.shape, conv2.shape, conv3.shape, conv4.shape)
        out = self.decoder(conv1, conv2, conv3, conv4)
        out_depth = self.last_layer_depth(out)
        print("****** Out_depth last_layer_depth: ", out_depth.shape)
        out_depth = torch.sigmoid(out_depth) * self.max_depth

        return {'pred_d': out_depth}

class adaptive_bins(nn.Module):
    def __init__(self, in_channels, embedding_dim=512, dim_out=256, norm = 'linear'):
        super().__init__()
        self.in_channels = in_channels
        # self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        self.dim_out = dim_out
        self.norm = norm

        self.conv_3x3 = nn.Conv2d(self.in_channels, self.embedding_dim, kernel_size=3, stride=1, padding=1)
        self.regressor = nn.Sequential(nn.Linear(self.embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, self.dim_out))

    def forward(self, x):
        #x the embeding sequence N, S, E turns to S, N, E
        x = x.permute(1, 0, 2)
        print("******* Shape of embed_sequence after permuting: ", x.shape)
        first_token = x[0, ...]
        print("******* Shape of first token of sequence: ", first_token.shape)
        first_token = self.regressor(first_token)
        # first_token = nn.Linear(512, 256)(first_token)
        print("******* Shape of first token after regressor: ", first_token.shape)

        # normalize the bins
        if self.norm == 'linear':
            b = torch.relu(first_token)
            eps = 0.1
            b = b + eps
        b = b / b.sum(dim=1, keepdim=True)
        print(b)
        return b

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.bot_conv = nn.Conv2d(
            in_channels=in_channels[0], out_channels=out_channels, kernel_size=1)
        

        self.skip_conv1 = nn.Conv2d(
            in_channels=in_channels[1], out_channels=out_channels, kernel_size=1)


        self.skip_conv2 = nn.Conv2d(
            in_channels=in_channels[2], out_channels=out_channels, kernel_size=1)


        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        print("****** Decoder channel-in = ", in_channels)
        print("****** Decoder channel-out = ", out_channels)
        
        self.fusion1 = SelectiveFeatureFusion(out_channels)
        self.fusion2 = SelectiveFeatureFusion(out_channels)
        self.fusion3 = SelectiveFeatureFusion(out_channels)

    def forward(self, x_1, x_2, x_3, x_4):
        x_4_ = self.bot_conv(x_4)
        out = self.up(x_4_)
        print("****** Out from bot_conv: ", x_4_.shape)
        print("****** Out from bot_conv after up: ", out.shape)

        x_3_ = self.skip_conv1(x_3)
        print("****** Out from skip_conv_1: ", x_3_.shape)
        out = self.fusion1(x_3_, out)
        out = self.up(out)

        x_2_ = self.skip_conv2(x_2)
        print("****** Out from skip_conv_2: ", x_2_.shape)

        out = self.fusion2(x_2_, out)
        out = self.up(out)

        out = self.fusion3(x_1, out)
        out = self.up(out)
        out = self.up(out)

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
    print("a")