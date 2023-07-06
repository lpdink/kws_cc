rm -rf hexagon_Debug_toolv85_v68
rm -rf android_Debug_aarch64

build_cmake hexagon BUILD=Debug DSP_ARCH=v68
build_cmake android BUILD=Debug

adb shell mkdir -p /data/local/tmp/speech_backend
adb push hexagon_Debug_toolv85_v68/ship/* /data/local/tmp/speech_backend
adb push android_Debug_aarch64/ship/* /data/local/tmp/speech_backend
adb push resources/* /data/local/tmp/speech_backend

adb shell chmod +x /data/local/tmp/speech_backend/*

adb shell 'cd /data/local/tmp/speech_backend/ ; export LD_LIBRARY_PATH=/data/local/tmp/speech_backend/; ADSP_LIBRARY_PATH="/data/local/tmp/speech_backend/"; DSP_LIBRARY_PATH="/data/local/tmp/speech_backend/";  ./speech_backend_android_exec'