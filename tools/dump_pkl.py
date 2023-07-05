import os
import sys
import torch
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
RES_PATH = os.path.join(ROOT, "../resources")
if not os.path.isdir(RES_PATH):
    os.mkdir(RES_PATH)


def main():
    pkl = torch.load(sys.argv[1])
    state_dict = pkl["state_dict"]
    weights_file = open(f"{RES_PATH}/weight.bin", "wb")
    with open(f"{RES_PATH}/graph.txt", "w") as file:
        for key, value in state_dict.items():
            if not key.endswith("tracked"):
                # 暂时不转储BN，TODO BN合并到conv中
                if not "bn" in str(key):
                    weights = value.cpu().numpy()
                    file.write(f"{key}\t{weights.dtype}\t{weights.shape}\n")
                    weights_file.write(memoryview(weights))
    weights_file.close()
    print(f"bin and graph write out to {RES_PATH}/")


if __name__ == "__main__":
    main()
