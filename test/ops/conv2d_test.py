import os
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.dirname(__file__))
TEST_TIMES=100

def compile():
    out_file = os.path.join(ROOT, "a.out")
    if not os.path.isfile(out_file):
        os.system(f"g++ {ROOT}/*.cc -o {ROOT}/a.out")

def main():
    # compile()
    with torch.no_grad():
        for _ in range(TEST_TIMES):
            py_conv2d = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=(5, 3), stride=(1, 2), padding=(2, 1), bias=True)
            # py_conv2d = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=2, padding=1, bias=True)
            input = torch.randn((1, 4, 128, 128))
            with open(f"{ROOT}/tensor.bin", "wb") as tensor_file:
                weight = memoryview(py_conv2d.weight.reshape(-1).numpy())
                bias = memoryview(py_conv2d.bias.reshape(-1).numpy())
                tensor_file.write(weight)
                tensor_file.write(bias)
            with open(f"{ROOT}/input.bin", "wb") as input_file:
                input = memoryview(input.reshape(-1).numpy())
                input_file.write(input)
            # os.system(f"{ROOT}/a.out {ROOT}/tensor.bin {ROOT}/input.bin")
            # 读取out.bin，验证与python的计算结果的相似性
        pass


if __name__=="__main__":
    main()