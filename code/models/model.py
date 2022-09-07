from turtle import forward
import torch
import torch.nn as nn

from mmcv.runner import load_checkpoint
from .mit import mit_b4

class GLPDepth(nn.Module):
    def __init__(self, max_depth=80.0, min_depth=0,is_train=False):
        super().__init__()
        self.max_depth = max_depth
        self.min_depth = min_depth
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

        self.ada_bins_4 = adaptive_bins(embedding_dim=512, dim_out=256, n_query_channels = 128, norm = 'linear')

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_out = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))
                                      
        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):      
        # print("****** Model input shape: ", x.shape)  
        # print("****** lenth of Encoder output: ", len(self.encoder(x)))  
        convs, embeds = self.encoder(x)

        #adabins section

        # print("****** Shape of embed_sequence 0 1 2 3  // {} {} {} {}: ", embeds[0].shape, embeds[1].shape, embeds[2].shape, embeds[3].shape) 
        conv1, conv2, conv3, conv4 = convs

        # b1, r1 = self.ada_bins(conv1 ,embeds[0])
        # b2, r2 = self.ada_bins(conv2 ,embeds[1])
        # b3, r3 = self.ada_bins(conv3 ,embeds[2])
        b4, range_attention_maps_4 = self.ada_bins_4(conv1 ,embeds[3])
        # print("Range attention map 4: ", range_attention_maps_4.shape)
        out = self.conv_out(range_attention_maps_4)
        # print("Range attention map 4 conv out: ", out.shape)
        out = self.up(out)
        out = self.up(out)
        # print("Range attention map 4 up x2: ", out.shape)

        bin_widths = (self.max_depth - self.min_depth) * b4  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_depth)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()

        centers = centers.view(n, dout, 1, 1)
        # print("****** centers ", centers.shape)
        out_depth = torch.sum(out * centers, dim=1, keepdim=True)

        return {'pred_d': out_depth}



        #original-flow of GLPDepth  
        # print("****** Output from the Encoder (mixtransformer): ", conv1.shape, conv2.shape, conv3.shape, conv4.shape)
        # out = self.decoder(conv1, conv2, conv3, conv4)
        # print("****** Output from the decoder: ", out.shape)
        # out_depth = self.last_layer_depth(out)
        # print("****** Out_depth last_layer_depth: ", out_depth.shape)
        # out_depth = torch.sigmoid(out_depth) * self.max_depth

        # return {'pred_d': out_depth}

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


class adaptive_bins(nn.Module):
    def __init__(self, embedding_dim=512, dim_out=256, n_query_channels = 64, norm = 'linear', ori_values_dim=64):
        super().__init__()
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        self.dim_out = dim_out
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.ori_values_dim = ori_values_dim

        # self.conv_3x3 = nn.Conv2d(self.in_channels, self.embedding_dim, kernel_size=3, stride=1, padding=1)
        self.regressor = nn.Sequential(nn.Linear(self.embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, self.dim_out))

        self.dot_product_layer = PixelWiseDotProduct(values_dim=self.embedding_dim, embedding_dim=self.ori_values_dim)
    def forward(self, x, y):
        #x: N, C, H, W
        #y: N, S, E

        embed = y.clone() #y has shape N, S, E turns to S, N, E
        embed = embed.permute(1, 0, 2)
        # print("******* Shape of embed_sequence after permuting: ", x.shape)
        first_token = embed[0, ...]
        queries = embed[1:self.n_query_channels + 1, ...]
        # print("Shape of first token: {} // shape of query: {}".format(first_token.shape, queries.shape))
        # print("******* Shape of first token of sequence: ", first_token.shape)
        y = self.regressor(first_token)
        # print("******* Shape of first token after regressor: ", y.shape)
        queries = queries.permute(1,0,2)
        range_attention_maps = self.dot_product_layer(x, queries)
        # print("range attention map: ", range_attention_maps.shape)

        # normalize the bins
        if self.norm == 'linear':
            y = torch.relu(y)
            eps = 0.1
            y = y + eps
        elif self.norm == 'softmax':
            return torch.softmax(y, dim=1), range_attention_maps
        else:
            y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True)
        # print(y)
        return y,range_attention_maps

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
        # print("****** Decoder channel-in = ", in_channels)
        # print("****** Decoder channel-out = ", out_channels)
        
        self.fusion1 = SelectiveFeatureFusion(out_channels)
        self.fusion2 = SelectiveFeatureFusion(out_channels)
        self.fusion3 = SelectiveFeatureFusion(out_channels)

    def forward(self, x_1, x_2, x_3, x_4):
        x_4_ = self.bot_conv(x_4)
        out = self.up(x_4_)
        # print("****** Out from bot_conv: ", x_4_.shape)
        # print("****** Out from bot_conv after up: ", out.shape)

        x_3_ = self.skip_conv1(x_3)
        # print("****** Out from skip_conv_1: ", x_3_.shape)
        out = self.fusion1(x_3_, out)
        out = self.up(out)

        x_2_ = self.skip_conv2(x_2)
        # print("****** Out from skip_conv_2: ", x_2_.shape)

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
    # print("a")