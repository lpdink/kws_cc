import os
import onnx
import argparse

from onnx import numpy_helper

ROOT = os.path.abspath(os.path.dirname(__file__))
DEFAULT_MODEL_NAME = "nn_4mic_4vocal"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default=f"{ROOT}/../resources/{DEFAULT_MODEL_NAME}.onnx",required=False,help="input onnx model path")
    parser.add_argument("-o", "--output", default=f"{ROOT}/../resources/", required=False, help="output folder path")

    args = parser.parse_args()
    if not os.path.isdir(args.output):
        raise NotADirectoryError(f"{args.output}")
    
    model = onnx.load(args.input)
    weights_file = open()
    with open("./graph.txt", "w") as file:
        for node in model.graph.initializer:
            weights = numpy_helper.to_array(node)
            file.write(f"{node.name}\t{weights.dtype}\t{weights.shape}\n")

if __name__=="__main__":
    main()