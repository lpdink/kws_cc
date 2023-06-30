# 定义本地路径
LOCAL_PATH := $(call my-dir)

# 清除之前的变量
include $(CLEAR_VARS)

# 定义模块名称
LOCAL_MODULE := crn_test

# 定义源文件
LOCAL_SRC_FILES := ${LOCAL_PATH}/src/*.cc

LOCAL_CPP_EXTENSION := .cc

LOCAL_C_INCLUDES := ${LOCAL_PATH}/inc/

# 定义目标架构
LOCAL_ARM_MODE := arm

# 定义目标平台
APP_PLATFORM := android-28

LOCAL_CFLAGS += -O3
APP_OPTIM := release


# 构建可执行文件
include $(BUILD_EXECUTABLE)

