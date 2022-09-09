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
        convs, embeds = self.encoder(x)
        conv1, conv2, conv3, conv4 = convs

        out = self.decoder(conv1, conv2, conv3, conv4)

        b4, range_attention_maps_4 = self.ada_bins_4(out ,embeds[3]) #param 1: the tensor to calculate R, param 2: to calculate b
        # b4: (N, dim_out)
        # range_attention_maps_4: (N, n_query_channels, H/4, W/4)

        out = self.conv_out(range_attention_maps_4) #convert the channel (n_query_channels) to the number of embedding bins: dim_out

        bin_widths = (self.max_depth - self.min_depth) * b4  # .shape = N, dim_out
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_depth)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)
        out_depth = torch.sum(out * centers, dim=1, keepdim=True)

        # print(out_depth)
        # out_depth = self.last_layer_depth(out)
        # out_depth = torch.sigmoid(out_depth) * self.max_depth

        return {'pred_d': out_depth}

class PixelWiseDotProduct(nn.Module):
    def __init__(self, values_dim=256, embedding_dim=128):
        super(PixelWiseDotProduct, self).__init__()
        self.values_dim = values_dim #change the dim of the values vector (encoder_1 = 64 to values_dim)
        self.embedding_dim = embedding_dim

        #conv1x1 in_channels: the channel of the volume itself, out_channels: the embeding_dim E
        self.conv1x1 = nn.Conv2d(in_channels=values_dim, out_channels=embedding_dim, kernel_size=1,stride=1)

    def forward(self, x, K):
        # X plays as the Key for the querry K: (N, C, H, W) H and W is depended on where X is selected in the architecture
        # K from the adative bins calculation: (N, n_query_channels, E)
        # Return: (N, n_query_channels, H, W)
        x = self.conv1x1(x) # conv1x1 to convert the volume X to has the channel number same as K => (N, E, H, W)
        n, c, h, w = x.size()
        _, cout, ck = K.size() #cout: equal to n_query_channels
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))  # .shape = n, hw, cout
        return y.permute(0, 2, 1).view(n, cout, h, w)


class adaptive_bins(nn.Module):
    def __init__(self, embedding_dim=512, dim_out=256, n_query_channels = 128, norm = 'linear'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dim_out = dim_out
        self.norm = norm
        self.n_query_channels = n_query_channels #number of tokens in embed sequence used for calculating R, ori adabin sets to 128

        # conv_3x3 to match the key volume X channels with the query K calculated
        # self.conv_3x3 = nn.Conv2d(self.in_channels, self.embedding_dim, kernel_size=3, stride=1, padding=1)

        #transform the b vector with the embedding_bim to dim_out = N_bins (number of devided bins of depth)
        self.regressor = nn.Sequential(nn.Linear(self.embedding_dim, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, self.dim_out))

        #the multiple the range_attention_map with the bins vector
        self.dot_product_layer = PixelWiseDotProduct(values_dim=64, embedding_dim=self.embedding_dim)

    def forward(self, x, y):
        """_summary_
        Args:
            x (tensor N C H W ): the block is multiplied with R vector later for calculating the range-attention-map
            y (tensor N S E): the ANY sequence of patches, for calculating b and R
        Returns:
            tensor: N nbins
            tensor: range-attention-map
        """
        #x: N, C, H, W
        #y: N, S, E

        # embed = y.clone() #y has shape N, S, E turns to S, N, 
        embed = y #testing case without .clone()
        embed = embed.permute(1, 0, 2)
        first_token = embed[0, ...]
        queries = embed[1:self.n_query_channels + 1, ...]
        y = self.regressor(first_token)

        # change from S N E to N S E
        queries = queries.permute(1,0,2)
        range_attention_maps = self.dot_product_layer(x, queries)

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
        return y, range_attention_maps

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
     
        self.fusion1 = SelectiveFeatureFusion(out_channels)
        self.fusion2 = SelectiveFeatureFusion(out_channels)
        self.fusion3 = SelectiveFeatureFusion(out_channels)

    def forward(self, x_1, x_2, x_3, x_4):
        x_4_ = self.bot_conv(x_4)
        out = self.up(x_4_)

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