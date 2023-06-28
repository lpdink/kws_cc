import torch
import torch.nn as nn
import numpy as np
# from settings.jd_4mic_4vocal import *

from thop import profile
from thop import clever_format

def numParams(net):
    count = sum([int(np.prod(param.shape)) for param in net.parameters()])
    return count

class SpeechConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, is_bn=True):
        super(SpeechConv, self).__init__()
        self.ln = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, bias=True)
        self.ln_bn = nn.BatchNorm2d(out_channels)
        self.gate = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=True)
        self.gate_bn = nn.BatchNorm2d(out_channels)
        self.is_bn = is_bn

    def forward(self, in_feat):
        ln = self.ln(in_feat)
        if self.is_bn:
            ln = self.ln_bn(ln)
            gate = torch.sigmoid(self.gate_bn(self.gate(in_feat)))
        else:
            gate = torch.sigmoid(self.gate(in_feat))
        res = ln * gate
        return res


class SpeechDeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, is_bn=True):
        super(SpeechDeConv, self).__init__()
        self.ln = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, bias=True)
        self.gate = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=True)
        self.is_bn = is_bn
        if is_bn:
            self.ln_bn = nn.BatchNorm2d(out_channels)
            self.gate_bn = nn.BatchNorm2d(out_channels)

    def forward(self, in_feat):
        ln = self.ln(in_feat)
        if self.is_bn:
            ln = self.ln_bn(ln)
            gate = torch.sigmoid(self.gate_bn(self.gate(in_feat)))
        else:
            gate = torch.sigmoid(self.gate(in_feat))
        res = ln * gate
        return res


class NetCRNNMask16ms(nn.Module):
    def __init__(self):
        super(NetCRNNMask16ms, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.mag_conv = SpeechConv(in_channels=4, out_channels=10, kernel_size=(5, 3), stride=(1, 2), padding=(2, 1))
        self.angle_conv = SpeechConv(in_channels=4, out_channels=10, kernel_size=(5, 3), stride=(1, 2), padding=(2, 1))

        self.conv2 = SpeechConv(in_channels=20, out_channels=20, kernel_size=(1, 3), stride=(1, 2),
                                padding=(0, 1))

        self.conv3 = SpeechConv(in_channels=20, out_channels=20, kernel_size=(1, 3), stride=(1, 2),
                                padding=(0, 1))

        self.conv4 = SpeechConv(in_channels=20, out_channels=20, kernel_size=(1, 3), stride=(1, 2),
                                padding=(0, 1))

        self.conv5 = SpeechConv(in_channels=20, out_channels=20, kernel_size=(1, 3), stride=(1, 2),
                                padding=(0, 1))

        self.conv6 = SpeechConv(in_channels=20, out_channels=20, kernel_size=(1, 3), stride=(1, 2),
                                padding=(0, 1))

        self.conv7 = SpeechConv(in_channels=20, out_channels=20, kernel_size=(1, 3), stride=(1, 2),
                                padding=(0, 1))

        self.lstm = nn.LSTM(input_size=60, hidden_size=60, num_layers=2, batch_first=True)

        # LSTM 初始化
        nn.init.orthogonal_(self.lstm.all_weights[0][0])
        nn.init.orthogonal_(self.lstm.all_weights[0][1])
        nn.init.zeros_(self.lstm.all_weights[0][2])
        nn.init.zeros_(self.lstm.all_weights[0][3])

        nn.init.orthogonal_(self.lstm.all_weights[1][0])
        nn.init.orthogonal_(self.lstm.all_weights[1][1])
        nn.init.zeros_(self.lstm.all_weights[1][2])
        nn.init.zeros_(self.lstm.all_weights[1][3])

        self.conv7_t = SpeechDeConv(in_channels=40, out_channels=20, kernel_size=(1, 3), stride=(1, 2),
                                    padding=(0, 1))

        self.conv6_t = SpeechDeConv(in_channels=40, out_channels=20, kernel_size=(1, 3), stride=(1, 2),
                                    padding=(0, 1))

        self.conv5_t = SpeechDeConv(in_channels=40, out_channels=20, kernel_size=(1, 3), stride=(1, 2),
                                    padding=(0, 1))

        self.conv4_t = SpeechDeConv(in_channels=40, out_channels=20, kernel_size=(1, 3), stride=(1, 2),
                                    padding=(0, 1))

        self.conv3_t = SpeechDeConv(in_channels=40, out_channels=20, kernel_size=(1, 3), stride=(1, 2),
                                    padding=(0, 1))

        self.conv2_t = SpeechDeConv(in_channels=40, out_channels=20, kernel_size=(1, 3), stride=(1, 2),
                                    padding=(0, 1))

        self.conv1_t = SpeechDeConv(in_channels=40, out_channels=16, kernel_size=(1, 3), stride=(1, 2),
                                    padding=(0, 1))

        self.conv_mag_out = SpeechConv(in_channels=20, out_channels=4, kernel_size=(1, 3), stride=(1, 1), 
                                   padding=(0, 1), is_bn=False)
        self.conv_mask_out = SpeechConv(in_channels=20, out_channels=12, kernel_size=(1, 3), stride=(1, 1), 
                                   padding=(0, 1), is_bn=False)


    def forward(self, mix_log_mag, mix_angle, hidden=None):
        
        b, c, frames, feat_dim = mix_log_mag.shape

        e1_mag   = self.mag_conv(mix_log_mag)
        e1_angle = self.angle_conv(mix_angle)
        e1 = torch.cat([e1_mag, e1_angle], dim=1)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)
        e7 = self.conv7(e6)

        lstm_in = e7.permute(0, 2, 1, 3).reshape(b, frames, -1)
        lstm_out, new_hidden = self.lstm(lstm_in, hidden)
        dcnn_in = lstm_out.reshape(b, frames, e7.size(1), -1).permute(0, 2, 1, 3)

        d7 = self.conv7_t(torch.cat([dcnn_in, e7], dim=1))
        d6 = self.conv6_t(torch.cat([d7, e6], dim=1))
        d5 = self.conv5_t(torch.cat([d6, e5], dim=1))
        d4 = self.conv4_t(torch.cat([d5, e4], dim=1))
        d3 = self.conv3_t(torch.cat([d4, e3], dim=1))
        d2 = self.conv2_t(torch.cat([d3, e2], dim=1))
        d1 = self.conv1_t(torch.cat([d2, e1], dim=1))  # [batch, 16, 623, 257]

        est_mag_main = self.relu(self.conv_mag_out(torch.cat([d1, mix_log_mag], dim=1))) # [batch, c, t, f]
        est_mask = self.sigmoid(self.conv_mask_out(torch.cat([d1, mix_log_mag], dim=1))) # [batch, c, t, f]

        return est_mag_main, est_mask, (new_hidden[0].detach(), new_hidden[1].detach())


if __name__ == '__main__':
    net = NetCRNNMask16ms()  # 109K
    param_size = 0
    for v in filter(lambda p: p.requires_grad, net.parameters()):
        v = v.detach().cpu().numpy().size
        param_size += v
    print('model param size is {}k'.format(param_size // 1024))
    print('model param size is {}kB'.format(param_size * 4 // 1024))

    mix = torch.randn((1, 4, 61, 257), dtype=torch.float32)

    macs, params = profile(net, inputs=(mix, mix, None))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)
