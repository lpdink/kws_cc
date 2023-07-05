import os
import torch
import torch.nn as nn
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
TEST_TIMES = 100

COMPILE_FLAG = "-g -O0 -std=c++11"


def compile():
    out_file = os.path.join(ROOT, "a.out")
    os.system(f"rm {out_file}")
    if not os.path.isfile(out_file):
        os.system(f"g++ {COMPILE_FLAG} {ROOT}/*.cc -I {ROOT}/../../inc -o {ROOT}/a.out")


def calculate_cosine_similarity(a1, a2):
    dot_product = np.dot(a1, a2)
    norm_a1 = np.linalg.norm(a1)
    norm_a2 = np.linalg.norm(a2)
    cosine_similarity = dot_product / (norm_a1 * norm_a2)
    return cosine_similarity


def diff_array(a1, a2):
    # print(f"mean:{a1.mean()}\t{a2.mean()}")
    # print(f"std:{a1.std()}\t{a2.std()}")
    # print(f"max:{a1.max()}\t{a2.max()}")
    # print(f"min:{a1.min()}\t{a2.min()}")
    print(f"sim: {calculate_cosine_similarity(a1, a2)}")


def main():
    compile()
    with torch.no_grad():
        for _ in range(TEST_TIMES):
            py_conv2d = nn.Conv2d(
                in_channels=4,
                out_channels=10,
                kernel_size=(5, 3),
                stride=(1, 2),
                padding=(2, 1),
                bias=True,
            )
            input = torch.randn((1, 4, 10, 10))
            py_rst = py_conv2d(input)
            with open(f"{ROOT}/tensor.bin", "wb") as tensor_file:
                weight = memoryview(py_conv2d.weight.reshape(-1).numpy())
                bias = memoryview(py_conv2d.bias.reshape(-1).numpy())
                tensor_file.write(weight)
                tensor_file.write(bias)
            with open(f"{ROOT}/input.bin", "wb") as input_file:
                input = memoryview(input.reshape(-1).numpy())
                input_file.write(input)
            with open(f"{ROOT}/py_rst.bin", "wb") as input_file:
                input = memoryview(py_rst.reshape(-1).numpy())
                input_file.write(input)
            os.system(f"{ROOT}/a.out")
            # 读取out.bin，验证与python的计算结果的相似性
            py_rst = np.fromfile("py_rst.bin", np.float32)
            cc_rst = np.fromfile("cc_rst.bin", np.float32)
            diff_array(py_rst, cc_rst)


if __name__ == "__main__":
    main()
