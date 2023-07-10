import torch
import torch.nn as nn
import numpy as np

in_channels = 4
out_chanels = 10
kernel_size = (5, 3)
stride = (1, 2)
# stride = (3, 3)
padding = (2, 1)
# padding = (2, 2)


def im2col(input: torch.Tensor):
    batch_size, channel, height, width = input.shape
    conv2d_rst_height = (height + padding[0] * 2 - kernel_size[0]) / stride[0] + 1
    conv2d_rst_width = (width + padding[1] * 2 - kernel_size[1]) / stride[1] + 1
    # 结果矩阵：
    # 行：卷积核扫过的次数，完成多少次点乘，等于输出的长宽之积
    # 列：一个卷积核内的所有元素，包含通道
    rst_row = int(conv2d_rst_height * conv2d_rst_width)
    rst_col = int(in_channels * kernel_size[0] * kernel_size[1])
    rst = torch.zeros((rst_col, rst_row))
    for r_idx in range(rst_row):
        # 计算窗口位置
        w_x = int(r_idx // conv2d_rst_width * stride[0])
        w_y = int(r_idx % conv2d_rst_width * stride[1])
        for c_idx in range(rst_col):
            # channel_idx = int((r_idx*rst_col+c_idx)/(kernel_size[0]*kernel_size[1])%channel)
            channel_idx = int(c_idx // (kernel_size[0] * kernel_size[1]))
            # 计算已经填充了的之前的channel的数量
            filled_channel_num = channel_idx * kernel_size[0] * kernel_size[1]
            x = int(w_x + (c_idx - filled_channel_num) // kernel_size[1]) - padding[0]
            y = int(w_y + (c_idx - filled_channel_num) % kernel_size[1]) - padding[1]
            if x < 0 or y < 0 or x >= height or y >= width:
                rst[c_idx, r_idx] = 0
            else:
                rst[c_idx, r_idx] = input[0, channel_idx, x, y]
    return rst


def calculate_cosine_similarity(a1, a2):
    dot_product = np.dot(a1, a2)
    norm_a1 = np.linalg.norm(a1)
    norm_a2 = np.linalg.norm(a2)
    cosine_similarity = dot_product / (norm_a1 * norm_a2)
    return cosine_similarity


def main():
    with torch.no_grad():
        layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_chanels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        weight = layer.weight.reshape(
            (out_chanels, in_channels * kernel_size[0] * kernel_size[1])
        )
        bias = layer.bias
        input = torch.randn((1, 4, 61, 257))
        im2col_rst = im2col(input)
        torch_rst = layer(input).reshape(-1)
        self_rst = torch.matmul(weight, im2col_rst)
        self_rst += torch.stack([bias] * im2col_rst.size(1), dim=1)
        self_rst = self_rst.reshape(-1)
        # print(torch_rst)
        # print(self_rst)
        print(calculate_cosine_similarity(torch_rst.numpy(), self_rst.numpy()))
        ## 由于浮点运算精度问题，不能使用==比较。
        # if torch.all(torch_rst==self_rst):
        #     print("PASSSSSSSSSSSSSSSSSSSSS!")
        # else:
        #     print("failed.")


if __name__ == "__main__":
    for i in range(100):
        main()
