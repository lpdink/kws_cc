# SpeechBackend

## Usage:

1. python tools/dump_pkl.py YOUR_PKL_PATH
2. sh build.sh
3. ./build/crn_test ./resources/weight.bin ./resources/weight.bin 
4. 这将读取权重，打印的offset应当恰为模型文件的4倍

## op test:

下列命令会完成编译、权重和输入的生成，结果写出、和py及cc结果对比的全部步骤。

```
cd test/ops/
python conv2d_test.py
```