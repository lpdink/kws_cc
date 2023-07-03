import numpy as np

class Conv2d:
    def __init__(self, in_channel=4, out_channel=10, kernel_size=(5, 3), stride=(1,2), padding=(2,1), use_bias=True) -> None:
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = use_bias
        self.weights = np.random.randn(out_channel, in_channel, *kernel_size)
        self.bias = np.random.randn(out_channel)

    def forward(self, x):
        ...
