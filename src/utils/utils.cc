#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

float* load_file(const char *file_path){
    int ret = -1, size=0;
    struct stat st;
    float *rst=nullptr;
    ret = stat(file_path, &st);
    if(ret==1){
        printf("file %s open failed.", file_path);
        exit(0);
    }
    else{
        size = st.st_size;
        FILE *file = fopen(file_path, "rb");
        rst = new float[size];
        fread(rst, sizeof(float), size, file);
        fclose(file);
    }
    return rst;
}