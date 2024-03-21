import pywt
import torch
from torch import nn


# CNN-Layer
class SConv_1D(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, kernel, pading, dilation=1, if_relu=True):
        super(SConv_1D, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pading, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU() if if_relu else nn.Sequential()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# WRT(Wavelet Reserve Transform)
class WRT(nn.Module):
    def __init__(self, wavelet="db16", dilation=1, stride=1):
        super(WRT, self).__init__()
        self.wavelet = self._get_wavelet(wavelet)
        self.dec_lo, self.dec_hi = self._prepare_filters()
        self.pad_lo, self.pad_hi = self._prepare_padding(dilation)
        self.conv_lo, self.conv_hi = self._prepare_convolutions(dilation, stride)

    def _get_wavelet(self, wavelet):
        """Prepare wavelet"""
        if isinstance(wavelet, pywt.Wavelet):
            return wavelet
        elif isinstance(wavelet, str):
            return pywt.Wavelet(wavelet)
        else:
            raise ValueError("wavelet must be a str or Wavelet object")

    def _prepare_filters(self):
        """Prepare filters"""
        dec_lo = torch.Tensor(self.wavelet.dec_lo).unsqueeze(0).unsqueeze(0)
        dec_hi = torch.Tensor(self.wavelet.dec_hi).unsqueeze(0).unsqueeze(0)
        return dec_lo, dec_hi

    def _prepare_padding(self, dilation):
        """Prepare padding"""
        padding_size_lo = self.dec_lo.size(-1) + (self.dec_lo.size(-1) - 1) * (dilation - 1)
        padding_size_hi = self.dec_hi.size(-1) + (self.dec_hi.size(-1) - 1) * (dilation - 1)
        pad_lo = nn.ZeroPad2d((padding_size_lo // 2, padding_size_lo // 2 - 1 + padding_size_lo % 2, 0, 0))
        pad_hi = nn.ZeroPad2d((padding_size_hi // 2, padding_size_hi // 2 - 1 + padding_size_hi % 2, 0, 0))
        return pad_lo, pad_hi

    def _prepare_convolutions(self, dilation, stride):
        """Prepare convolutions"""
        conv_lo = nn.Conv1d(1, 1, kernel_size=self.dec_lo.size(-1), bias=False, dilation=dilation, stride=stride)
        conv_hi = nn.Conv1d(1, 1, kernel_size=self.dec_hi.size(-1), bias=False, dilation=dilation, stride=stride)
        conv_lo.weight.data = self.dec_lo
        conv_hi.weight.data = self.dec_hi
        conv_lo.weight.requires_grad = False
        conv_hi.weight.requires_grad = False
        return conv_lo, conv_hi

    def forward(self, x):
        batch_size, num_channels, signal_length = x.size()
        x_flat = x.view(batch_size * num_channels, 1, signal_length)
        cA1_flat = self.conv_lo(self.pad_lo(x_flat))
        cD1_flat = self.conv_hi(self.pad_hi(x_flat))
        cA1 = cA1_flat.view(batch_size, num_channels, -1)
        cD1 = cD1_flat.view(batch_size, num_channels, -1)
        return cA1, cD1


# MWR(Multi-scale Wavelet Reserve)
class MWR(nn.Module):
    def __init__(self, waves=None, dilation=1, stride=1):
        super(MWR, self).__init__()
        if waves is None:
            waves = ['db4', 'db16']

        # Initialize the wavelet network, one network per wavelet
        self.wavelet_nets = nn.ModuleList([WRT(wavelet, dilation=dilation, stride=stride) for wavelet in waves])

    def forward(self, x):
        """The input signal is processed and the wavelet transform results of different scales are combined"""
        # Processing wavelet networks
        coefficients = [wavelet_net(x) for wavelet_net in self.wavelet_nets]
        # The approximate coefficient and the detail coefficient are combined respectively
        approx_coeffs, detail_coeffs = zip(*coefficients)
        combined_approx = torch.cat(approx_coeffs, dim=1)
        combined_detail = torch.cat(detail_coeffs, dim=1)
        # concatenate and return
        return torch.cat([combined_approx, combined_detail], dim=1)


class MWRC4_ResNet10(nn.Module):
    def __init__(self, num_classes=9, input=3, numf=12):
        super(MWRC4_ResNet10, self).__init__()
        self.relu = nn.ReLU()

        self.mwr0 = MWR(['db4', 'db8', 'db12', 'db16'])
        self.sconv0 = SConv_1D(input * 8, numf * 4, 3, 1)

        self.mwr1 = MWR(['db4', 'db8', 'db12', 'db16'])
        self.sconv1 = SConv_1D(numf * 32, numf * 4, 3, 1, if_relu=False)

        self.mwr2 = MWR(['db4', 'db8', 'db12', 'db16'])
        self.sconv2 = SConv_1D(numf * 32, numf * 4, 3, 1, if_relu=False)

        self.mwr3 = MWR(['db4', 'db8', 'db12', 'db16'], stride=2)
        self.Maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.sconv3 = SConv_1D(numf * 32, numf * 4, 3, 1, if_relu=False)

        self.mwr4 = MWR(['db4', 'db8', 'db12', 'db16'])
        self.sconv4 = SConv_1D(numf * 32, numf * 4, 3, 1, if_relu=False)

        self.mwr5 = MWR(['db4', 'db8', 'db12', 'db16'])
        self.sconv5 = SConv_1D(numf * 32, numf * 4, 3, 1, if_relu=False)

        self.mwr6 = MWR(['db4', 'db8', 'db12', 'db16'], stride=2)
        self.Maxpool6 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.sconv6 = SConv_1D(numf * 32, numf * 4, 3, 1, if_relu=False)

        self.mwr7 = MWR(['db4', 'db8', 'db12', 'db16'])
        self.sconv7 = SConv_1D(numf * 32, numf * 4, 3, 1, if_relu=False)

        self.mwr8 = MWR(['db4', 'db8', 'db12', 'db16'])
        self.sconv8 = SConv_1D(numf * 32, numf * 4, 3, 1, if_relu=False)

        self.mwr9 = MWR(['db4', 'db8', 'db12', 'db16'], stride=2)
        self.Maxpool9 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.sconv9 = SConv_1D(numf * 32, numf * 4, 3, 1, if_relu=False)

        self.sconv10 = SConv_1D(numf * 4, numf * 4, 3, 1)

        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(numf * 4, num_classes)

    def forward(self, inputs):
        output = self.mwr0(inputs)
        output0 = self.sconv0(output)

        output = self.mwr1(output0)
        output0 = self.relu(self.sconv1(output) + output0)

        output = self.mwr2(output0)
        output0 = self.relu(self.sconv2(output) + output0)

        output = self.mwr3(output0)
        output0 = self.relu(self.sconv3(output) + self.Maxpool3(output0))

        output = self.mwr4(output0)
        output0 = self.relu(self.sconv4(output) + output0)

        output = self.mwr5(output0)
        output0 = self.relu(self.sconv5(output) + output0)

        output = self.mwr6(output0)
        output0 = self.relu(self.sconv6(output) + self.Maxpool6(output0))

        output = self.mwr7(output0)
        output0 = self.relu(self.sconv7(output) + output0)

        output = self.mwr8(output0)
        output0 = self.relu(self.sconv8(output) + output0)

        output = self.mwr9(output0)
        output0 = self.relu(self.sconv9(output) + self.Maxpool9(output0))

        output = self.sconv10(output0)

        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


if __name__ == '__main__':
    inputs = torch.ones(32, 3, 2048)  # Example 1D signal
    model = MWRC4_ResNet10(9, numf=20)
    output = model(inputs)
    print(output.shape)
