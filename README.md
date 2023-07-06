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

## test on hexagon:

build.sh脚本包含了向hexagon终端推送编译产物和所需资源文件的逻辑，去掉注释即可。

```
source ${hexagon_sdk}/setup_sdk_env.source
cd hexagon
sh build.sh
```