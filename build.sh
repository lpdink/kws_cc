root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
rm -rf $root/build
mkdir -p $root/build

# build with ndk
ndk-build NDK_PROJECT_PATH="./" APP_BUILD_SCRIPT="./Android.mk" NDK_LIBS_OUT=./build/libs NDK_OUT=./build/obj

# build with cmake
cd $root/build 
cmake ..
make