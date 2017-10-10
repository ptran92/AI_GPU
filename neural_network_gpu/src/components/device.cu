/*************************************************************
*   File: device.cu
*
*
*************************************************************/
#include "device.h"

/*************************************************************
 *    STATIC VARIABLES
 *************************************************************/
cublasHandle_t Device::cublas_hdl;
const int      Device::total_threads;

/*************************************************************
 *    PUBLIC FUNCTIONS
 *************************************************************/
void Device::Device_Create(void)
{
  cublasCreate(&cublas_hdl);
}

cublasHandle_t Device::Device_Get_Handle(void)
{
  return cublas_hdl;
}

void Device::Device_Destroy(void)
{
  cublasDestroy(cublas_hdl);
}
