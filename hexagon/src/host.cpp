/*==============================================================================
  Copyright (c) 2015, 2020-2021 Qualcomm Technologies, Inc.
  All rights reserved. Qualcomm Proprietary and Confidential.
==============================================================================*/
#include "AEEStdErr.h"
#include "speech_backend.h"
#include "rpcmem.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "os_defines.h"
#include "verify.h"
#include "dsp_capabilities_utils.h"     // available under <HEXAGON_SDK_ROOT>/utils/examples
#include "os_defines.h"
#include <string.h>

#define MAX_LINE_LEN (100)

static void print_usage()
{
    printf( "Usage:\n"
            "Options:\n"
            "-d domain: Run on a specific domain.\n"
            "    0: Run the example on ADSP\n"
            "    3: Run the example on CDSP\n"
            "    2: Run the example on SDSP\n"
            "        Default Value: 3(CDSP) for targets having CDSP and 0(ADSP) for targets not having CDSP like Agatti.\n"
            "-U unsigned_PD: Run on signed or unsigned PD.\n"
            "    0: Run on signed PD.\n"
            "    1: Run on unsigned PD.\n"
            "        Default Value: 1\n"
            );
}

int call_skel(int domain_id, bool is_unsignedpd_enabled)
{
    int nErr = AEE_SUCCESS;

    int speech_backend_URI_domain_len = strlen(speech_backend_URI) + MAX_DOMAIN_URI_SIZE;
    char *speech_backend_URI_domain = NULL;
    remote_handle64 handle = -1;
    int retVal = 0;
    int heapid = RPCMEM_HEAP_ID_SYSTEM;
    domain *my_domain = NULL;

#if defined(SLPI) || defined(MDSP)
    heapid = RPCMEM_HEAP_ID_CONTIG;
#endif

    // rpcmem_init();

    my_domain = get_domain(domain_id);
    if (my_domain == NULL) {
        nErr = AEE_EBADPARM;
        printf("\nERROR 0x%x: unable to get domain struct %d\n", nErr, domain_id);
    }

    if(is_unsignedpd_enabled) {
        if(remote_session_control) {
            struct remote_rpc_control_unsigned_module data;
            data.domain = domain_id;
            data.enable = 1;
            if (AEE_SUCCESS != (retVal = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void*)&data, sizeof(data)))) {
                printf("ERROR 0x%x: remote_session_control failed\n", retVal);
            }
        }
        else {
            retVal = AEE_EUNSUPPORTED;
            printf("ERROR 0x%x: remote_session_control interface is not supported on this device\n", retVal);
        }
    }

    if ((speech_backend_URI_domain = (char *)malloc(speech_backend_URI_domain_len)) == NULL) {
        nErr = AEE_ENOMEMORY;
        printf("unable to allocated memory for speech_backend_URI_domain of size: %d", speech_backend_URI_domain_len);
    }

    nErr = snprintf(speech_backend_URI_domain, speech_backend_URI_domain_len, "%s%s", speech_backend_URI, my_domain->uri);
    if (nErr < 0) {
        printf("ERROR 0x%x returned from snprintf\n", nErr);
        nErr = AEE_EFAILED;
    }

    retVal = speech_backend_open(speech_backend_URI_domain, &handle);
    if (retVal != 0) {
        printf("ERROR 0x%x: Unable to create FastRPC session on domain %d\n", retVal, domain_id);
        printf("Exiting...\n");
    }

    // call function here.
    int num1=42;int num2=88;
    int rst = 0;
    if (AEE_SUCCESS == (nErr = speech_backend_plus(handle, num1, num2, &rst))) {
           printf("dsp rst:%d\n", rst);
        }

    if (AEE_SUCCESS == (nErr = speech_backend_conv2d(handle))) {
        printf("\nconv2d finished\n");
    }

    // free
    if (speech_backend_URI_domain) {
        free(speech_backend_URI_domain);
    }

    if (handle) {
        speech_backend_close(handle);
    }

    // rpcmem_deinit();

    return nErr;
}


int main(int argc, char *argv[])
{
    int nErr = AEE_SUCCESS;
    int option = 0;
    int domain_id = -1;
    int unsignedpd_flag = 1;
    bool is_unsignedpd_enabled = false;

    while((option = getopt(argc, argv,"d:m:i:U:")) != -1) {
        switch (option) {
            case 'd' : domain_id = atoi(optarg);
                break;
            case 'U' : unsignedpd_flag = atoi(optarg);
                break;
            default:
                print_usage();
                return AEE_EUNKNOWN;
        }
    }

    if(domain_id == -1) {
        printf("\nDSP domain is not provided. Retrieving DSP information using Remote APIs.\n");
        nErr = get_dsp_support(&domain_id);
        if(nErr != AEE_SUCCESS) {
            printf("ERROR in get_dsp_support: 0x%x, defaulting to CDSP domain\n", nErr);
        }
    }

    if (!is_valid_domain_id(domain_id, 0)) {
        nErr = AEE_EBADPARM;
        printf("\nERROR 0x%x: Invalid domain %d\n", nErr, domain_id);
        print_usage();
        goto bail;
    }

    if (unsignedpd_flag < 0 || unsignedpd_flag > 1) {
        nErr = AEE_EBADPARM;
        printf("\nERROR 0x%x: Invalid unsigned PD flag %d\n", nErr, unsignedpd_flag);
        print_usage();
        goto bail;
    }

    if(unsignedpd_flag == 1) {
        is_unsignedpd_enabled = is_unsignedpd_supported(domain_id);
        if (!is_unsignedpd_enabled) {
            printf("Overriding user request for unsigned PD. Only signed offload is allowed on domain %d.\n", domain_id);
            unsignedpd_flag = 0;
        }
    }


    printf("Attempting to run on %s PD on domain %d\n", is_unsignedpd_enabled==true?"unsigned":"signed", domain_id);
    nErr = call_skel(domain_id, is_unsignedpd_enabled);
    if (nErr) {
        printf("ERROR 0x%x: test failed\n\n", nErr);
    }

bail:
    if (nErr) {
        printf("ERROR 0x%x: Test FAILED\n", nErr);
    } else {
        printf("Success\n");
    }
    return nErr;
}
