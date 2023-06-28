import os
import torch
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
RES_PATH = os.path.join(ROOT, "../resources")



def main():
    pkl = torch.load(f"{RES_PATH}/model-51000--1.3523531392216683.pickle")
    state_dict = pkl["state_dict"]
    weights_file = open(f"{RES_PATH}/weight.bin", "wb")
    with open(f"{RES_PATH}/graph.txt", "w") as file: 
        for key, value in state_dict.items():
            if not key.endswith("tracked"):
                weights = value.cpu().numpy()
                file.write(f"{key}\t{weights.dtype}\t{weights.shape}\n")
                # breakpoint()
                weights_file.write(memoryview(weights))
    weights_file.close()

if __name__=="__main__":
    main()