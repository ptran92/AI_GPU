#ifndef __DEVICE_H
#define __DEVICE_H

#include "cuda_runtime.h"
#include "cublas_v2.h"

#define CUDA_BLOCKS(N)          ( (N + 255) / 256 )

class Device
{
public:
  static void           Device_Create(void);
  static cublasHandle_t Device_Get_Handle(void);
  static void           Device_Destroy(void);
  
public:
  static const int      total_threads = 256;

private:
  static cublasHandle_t cublas_hdl;

};

#endif
