import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils_tools import ImageBatchNormalization

class A_Net(nn.Module):
    def __init__(self, in_channels=32, out_channels=3):
        super(A_Net, self).__init__()
        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))  
        self._block3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self._block4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self._block5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        # Layers: enc_conv6, upsample5
        self._block6 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block7 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block8 = nn.Sequential(
            nn.Conv2d(64+32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block9 = nn.Sequential(
            nn.Conv2d(64+32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block10 = nn.Sequential(
            nn.Conv2d(64+32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block11 = nn.Sequential(
            nn.Conv2d(64 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Sigmoid()
            )
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """
        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block3(pool2)
        pool4 = self._block4(pool3)
        pool5 = self._block5(pool4)
        # Decoder
        upsample5 = self._block6(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block7(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block9(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block9(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block10(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        return self._block11(concat1)# , pool5


class N_Net(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(N_Net, self).__init__()
        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))  
        self._block3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self._block4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self._block5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        # Layers: enc_conv6, upsample5
        self._block6 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block7 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block8 = nn.Sequential(
            nn.Conv2d(64+32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block9 = nn.Sequential(
            nn.Conv2d(64+32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block10 = nn.Sequential(
            nn.Conv2d(64+32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block11 = nn.Sequential(
            nn.Conv2d(64 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Tanh()
            )
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """
        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block3(pool2)
        pool4 = self._block4(pool3)
        pool5 = self._block5(pool4)
        # Decoder
        upsample5 = self._block6(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block7(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block9(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block9(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block10(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        return self._block11(concat1) , pool5


class S_Net(nn.Module):
    def __init__(self):
        super(S_Net, self).__init__()
        self._block = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 3, 1, 1),
        )

    def forward(self, input):
        s1 = self._block(input)
        return s1#s5.expand([n, 3, w, h])

class L_Net(nn.Module):
    def __init__(self):
        super(L_Net, self).__init__()
        self.conv1 = get_conv(32, 32, kernel_size=1, stride=1)
        self.pool1 = nn.AvgPool2d(4, stride=1, padding=0)
        self.conv2 = get_conv(32, 32, kernel_size=1, stride=2)
        self.pool2 = nn.AvgPool2d(3, stride=1, padding=0)
        self.fc = nn.Linear(32 * 1 * 1, 9)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.pool1(out1)
        # reshape to batch_size x 128
        out3 = self.conv2(out2)
        out4 = self.pool2(out3)
        out5 = out4.view(-1, 32 * 1 * 1)
        out5 = self.fc(out5)
        # out6 = torch.tanh(out5)
        return out5

class HD_Net(nn.Module):
    def __init__(self, d=64):
        super(HD_Net, self).__init__()
        self.conv_1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.albedo_gen_model = A_Net()
        self.out_shading = S_Net()
        self.normal_gen_model = N_Net()
        self.light = L_Net()

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward_once(self, input):
        layer_1 = self.conv_1(input)
        pred_albedo = self.albedo_gen_model(layer_1)
        pred_shading = self.out_shading(layer_1)

        pred_normal, feature = self.normal_gen_model(pred_shading.detach())
        # pred_normal, feature = self.normal_gen_model(pred_shading)

        pred_lighting = self.light(feature)

        return pred_albedo, pred_shading, pred_normal, pred_lighting

    def forward(self, input_1, input_2):
        pA_1, pS_1, pN_1, pL_1 = self.forward_once(input_1)
        pA_2, pS_2, pN_2, pL_2 = self.forward_once(input_2)
        return pA_1, pS_1, pN_1, pL_1, pA_2, pS_2, pN_2, pL_2

    def fix_weights(self):
        dfs_freeze(self.conv_1)
        dfs_freeze(self.albedo_gen_model)
        dfs_freeze(self.out_shading)
        dfs_freeze(self.normal_gen_model)

        # Note that we are not freezing Albedo gen model





class All_SharedEncoder(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""
    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""
        super(All_SharedEncoder, self).__init__()
      # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))  
        self._block3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self._block4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self._block5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)
        return pool1, pool2, pool3, pool4, pool5 


class DecoderShading(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(DecoderShading, self).__init__()
        self._block6 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block7 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block8 = nn.Sequential(
            nn.Conv2d(64+32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block9 = nn.Sequential(
            nn.Conv2d(64+32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block10 = nn.Sequential(
            nn.Conv2d(64+32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block11 = nn.Sequential(
            nn.Conv2d(64 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Sigmoid()
            )
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, pool1, pool2, pool3, pool4, pool5):
        """Through encoder, then decoder by adding U-skip connections. """
        # Encoder
        
        # Decoder
        upsample5 = self._block6(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block7(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block9(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block9(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block10(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        return self._block11(concat1)# , pool5

class DecoderAlbedo(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(DecoderAlbedo, self).__init__()
        self._block6 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block7 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block8 = nn.Sequential(
            nn.Conv2d(64+32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block9 = nn.Sequential(
            nn.Conv2d(64+32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block10 = nn.Sequential(
            nn.Conv2d(64+32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block11 = nn.Sequential(
            nn.Conv2d(64 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Sigmoid()
            )
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, pool1, pool2, pool3, pool4, pool5):
        """Through encoder, then decoder by adding U-skip connections. """
        # Encoder
        
        # Decoder
        upsample5 = self._block6(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block7(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block9(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block9(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block10(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        return self._block11(concat1)# , pool5

class DecoderNormal(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(DecoderNormal, self).__init__()
        self._block6 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block7 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block8 = nn.Sequential(
            nn.Conv2d(64+32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block9 = nn.Sequential(
            nn.Conv2d(64+32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block10 = nn.Sequential(
            nn.Conv2d(64+32, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))
        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block11 = nn.Sequential(
            nn.Conv2d(64 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Tanh()
            )
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, pool1, pool2, pool3, pool4, pool5):
        """Through encoder, then decoder by adding U-skip connections. """
        # Encoder
        
        # Decoder
        upsample5 = self._block6(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block7(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block9(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block9(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block10(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        return self._block11(concat1)# , pool5


class HD_Half_SharedEncoder(nn.Module):
    def __init__(self, d=64):
        super(HD_Half_SharedEncoder, self).__init__()
        self.EncoderFeature1 = All_SharedEncoder()
        self.EncoderFeature2 = All_SharedEncoder()

        self.DecoderShading = DecoderShading()
        self.DecoderLighting = L_Net()
        self.DecoderAlbedo = DecoderAlbedo()
        self.DecoderNormal = DecoderNormal()

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward_once(self, input):
        Feat1, Feat2, Feat3, Feat4, Feat5 = self.EncoderFeature1(input)
        pred_shading = self.DecoderShading(input, Feat1, Feat2, Feat3, Feat4, Feat5)
        pred_albedo = self.DecoderAlbedo(input, Feat1, Feat2, Feat3, Feat4, Feat5)
        
        Seat1, Seat2, Seat3, Seat4, Seat5 = self.EncoderFeature2(pred_shading)
        pred_normal = self.DecoderNormal(pred_shading, Seat1, Seat2, Seat3, Seat4, Seat5)
        pred_lighting = self.DecoderLighting(Seat5)

        return pred_albedo, pred_shading, pred_normal, pred_lighting

    def forward(self, input_1, input_2):
        pA_1, pS_1, pN_1, pL_1 = self.forward_once(input_1)
        pA_2, pS_2, pN_2, pL_2 = self.forward_once(input_2)

        # rec_F_1 = self.get_face(pS_1, pA_1)
        # rec_F_2 = self.get_face(pS_2, pA_2)

        # rec_S_1 = self.get_shading(pN_1, pL_1)
        # rec_S_2 = self.get_shading(pN_2, pL_2)
        return pA_1, pS_1, pN_1, pL_1, pA_2, pS_2, pN_2, pL_2

    def fix_weights(self):
        dfs_freeze(self.conv_1)
        dfs_freeze(self.albedo_gen_model)
        dfs_freeze(self.out_shading)

# Use following to fix weights of the model
# Ref - https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/15
def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


class Gradient_Net(nn.Module):
    def __init__(self):
        super(Gradient_Net, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)

        kernel_y = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        gray_input = (x[:, 0, :, :] + x[:, 1, :, :] + x[:, 2, :, :]).reshape([x.size(0), 1, x.size(2), x.size(2)]) / 3
        grad_x = F.conv2d(gray_input, self.weight_x)
        grad_y = F.conv2d(gray_input, self.weight_y)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient

class Gradient_Net_Single(nn.Module):
    def __init__(self):
        super(Gradient_Net_Single, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)

        kernel_y = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        gray_input = x
        grad_x = F.conv2d(gray_input, self.weight_x)
        grad_y = F.conv2d(gray_input, self.weight_y)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient

##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

# Base methods for creating convnet
def get_conv(in_channels, out_channels, kernel_size=3, padding=0, stride=1, dropout=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout)
    )

att= torch.from_numpy(np.pi*np.array([1, 2.0/3.0, 1/4.0])).type(torch.FloatTensor)
aa0 = torch.from_numpy(np.array(0.5/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)
aa1 = torch.from_numpy(np.array(np.sqrt(3)/2/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)
aa2 = torch.from_numpy(np.array(np.sqrt(15)/2/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)
aa3 = torch.from_numpy(np.array(np.sqrt(5)/4/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)
aa4 = torch.from_numpy(np.array(np.sqrt(15)/4/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)


def get_shading_DPR_B(N, L):
    # N: DPR normal coordinate
    b, c, h, w = N.shape
    Norm = N.reshape([b, c, -1]).permute([0, 2, 1])  # b*c*65536   ----  b* 65536 * c
    norm_X = Norm[:, :, 2]
    norm_Y = Norm[:, :, 0]
    norm_Z = -Norm[:, :, 1]
    sh_basis = torch.from_numpy(np.zeros([b, h*w, 9])).type(torch.FloatTensor).to(N.device)
    sh_basis[:, :, 0] = aa0.to(N.device)*att[0]
    sh_basis[:, :, 1] = aa1.to(N.device)*norm_Y*att[1]
    sh_basis[:, :, 2] = aa1.to(N.device)*norm_Z*att[1]
    sh_basis[:, :, 3] = aa1.to(N.device)*norm_X*att[1]

    sh_basis[:, :, 4] = aa2.to(N.device)*norm_Y*norm_X*att[2]
    sh_basis[:, :, 5] = aa2.to(N.device)*norm_Y*norm_Z*att[2]
    sh_basis[:, :, 6] = aa3.to(N.device)*(3*norm_Z**2-1)*att[2]
    sh_basis[:, :, 7] = aa2.to(N.device)*norm_X*norm_Z*att[2]
    sh_basis[:, :, 8] = aa4.to(N.device)*(norm_X**2-norm_Y**2)*att[2]

    shading = torch.matmul(sh_basis, L.unsqueeze(2)).permute([0, 2, 1]).reshape([b, 1, w, h])
    shading = ImageBatchNormalization(shading)
    return shading
   
def get_L_DPR_B(S, N):
    b, c, h, w = N.shape
    Norm = N.reshape([b, c, -1]).permute([0, 2, 1])  # b*c*65536   ----  b* 65536 * c
    norm_X = Norm[:, :, 2]
    norm_Y = Norm[:, :, 0]
    norm_Z = -Norm[:, :, 1]
    sh_basis = torch.from_numpy(np.zeros([b, h*w, 9])).type(torch.FloatTensor).to(N.device)
    sh_basis[:, :, 0] = aa0.to(N.device)*att[0]
    sh_basis[:, :, 1] = aa1.to(N.device)*norm_Y*att[1]
    sh_basis[:, :, 2] = aa1.to(N.device)*norm_Z*att[1]
    sh_basis[:, :, 3] = aa1.to(N.device)*norm_X*att[1]

    sh_basis[:, :, 4] = aa2.to(N.device)*norm_Y*norm_X*att[2]
    sh_basis[:, :, 5] = aa2.to(N.device)*norm_Y*norm_Z*att[2]
    sh_basis[:, :, 6] = aa3.to(N.device)*(3*norm_Z**2-1)*att[2]
    sh_basis[:, :, 7] = aa2.to(N.device)*norm_X*norm_Z*att[2]
    sh_basis[:, :, 8] = aa4.to(N.device)*(norm_X**2-norm_Y**2)*att[2]

    S = S[:, 0, :, :].reshape([b, -1]).unsqueeze(2)
    com_L = torch.bmm(torch.pinverse(sh_basis), S).squeeze(2)

    return com_L


att= torch.from_numpy(np.pi*np.array([1, 2.0/3.0, 1/4.0])).type(torch.FloatTensor)
aa0 = torch.from_numpy(np.array(0.5/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)
aa1 = torch.from_numpy(np.array(np.sqrt(3)/2/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)
aa2 = torch.from_numpy(np.array(np.sqrt(15)/2/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)
aa3 = torch.from_numpy(np.array(np.sqrt(5)/4/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)
aa4 = torch.from_numpy(np.array(np.sqrt(15)/4/np.sqrt(np.pi),dtype=float)).type(torch.FloatTensor)


def get_shading_DPR_B(N, L, VisLight=True):
    # N: DPR normal coordinate
    b, c, h, w = N.shape
    Norm = N.reshape([b, c, -1]).permute([0, 2, 1])  # b*c*65536   ----  b* 65536 * c
    if VisLight == True:
        norm_X = Norm[:, :, 0]
        norm_Y = Norm[:, :, 1]
        norm_Z = Norm[:, :, 2]
    else:
        norm_X = Norm[:, :, 2]
        norm_Y = Norm[:, :, 0]
        norm_Z = -Norm[:, :, 1]
    sh_basis = torch.from_numpy(np.zeros([b, h*w, 9])).type(torch.FloatTensor).to(N.device)
    sh_basis[:, :, 0] = aa0.to(N.device)*att[0]
    sh_basis[:, :, 1] = aa1.to(N.device)*norm_Y*att[1]
    sh_basis[:, :, 2] = aa1.to(N.device)*norm_Z*att[1]
    sh_basis[:, :, 3] = aa1.to(N.device)*norm_X*att[1]

    sh_basis[:, :, 4] = aa2.to(N.device)*norm_Y*norm_X*att[2]
    sh_basis[:, :, 5] = aa2.to(N.device)*norm_Y*norm_Z*att[2]
    sh_basis[:, :, 6] = aa3.to(N.device)*(3*norm_Z**2-1)*att[2]
    sh_basis[:, :, 7] = aa2.to(N.device)*norm_X*norm_Z*att[2]
    sh_basis[:, :, 8] = aa4.to(N.device)*(norm_X**2-norm_Y**2)*att[2]

    shading = torch.matmul(sh_basis, L.unsqueeze(2)).permute([0, 2, 1]).reshape([b, 1, w, h])
    shading = ImageBatchNormalization(shading)
    return shading
   
def get_L_DPR_B(S, N):
    b, c, h, w = N.shape
    Norm = N.reshape([b, c, -1]).permute([0, 2, 1])  # b*c*65536   ----  b* 65536 * c

    norm_X = Norm[:, :, 2]
    norm_Y = Norm[:, :, 0]
    norm_Z = -Norm[:, :, 1]
    sh_basis = torch.from_numpy(np.zeros([b, h*w, 9])).type(torch.FloatTensor).to(N.device)
    sh_basis[:, :, 0] = aa0.to(N.device)*att[0]
    sh_basis[:, :, 1] = aa1.to(N.device)*norm_Y*att[1]
    sh_basis[:, :, 2] = aa1.to(N.device)*norm_Z*att[1]
    sh_basis[:, :, 3] = aa1.to(N.device)*norm_X*att[1]

    sh_basis[:, :, 4] = aa2.to(N.device)*norm_Y*norm_X*att[2]
    sh_basis[:, :, 5] = aa2.to(N.device)*norm_Y*norm_Z*att[2]
    sh_basis[:, :, 6] = aa3.to(N.device)*(3*norm_Z**2-1)*att[2]
    sh_basis[:, :, 7] = aa2.to(N.device)*norm_X*norm_Z*att[2]
    sh_basis[:, :, 8] = aa4.to(N.device)*(norm_X**2-norm_Y**2)*att[2]

    S = S[:, 0, :, :].reshape([b, -1]).unsqueeze(2)
    com_L = torch.bmm(torch.pinverse(sh_basis), S).squeeze(2)

    return com_L
