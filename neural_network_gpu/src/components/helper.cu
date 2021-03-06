/*************************************************************
*   File: helper.cu
*
*
*************************************************************/
#include <ctime>
#include <random>
#include "helper.h"
#include "cublas_v2.h"
#include "curand.h"
#include "curand_kernel.h"
#include "cuda_runtime.h"
#include "device.h"
#include "fp16_conversion.h"
/*************************************************************
 *    KERNEL FUNCTIONS
 *************************************************************/
/**************************************************************
 *
 *
 * SINGLE PRECISION FLOATING POINT KERNELS
 *
 *
 **************************************************************/
 __global__ void CrossEntropyLoss_Derivative_Gpu(const float * neural_out, const float * expect_out, float * loss_dvt, const int n)
 {
   int tid = blockIdx.x * blockDim.x + threadIdx.x;

   if(tid < n)
   {
     loss_dvt[tid] = -expect_out[tid] / neural_out[tid] + (1 - expect_out[tid]) / (1 - neural_out[tid]);
   }
 }

 __global__ void Sigmoid_Gpu(const float * z, float * output, const int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < n)
  {
    output[tid] = 1.0 / (1.0 + exp(-z[tid]));
  }
}

__global__ void Sigmoid_Dev_Gpu(const float * output, float * act_dvt, const int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < n)
  {
    act_dvt[tid] = output[tid] * (1 - output[tid]);
  }
}

__global__ void Err_Dev_Gpu(const float * error_signal, const float * act_dvt, float * err_dvt, const int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < n)
  {
    err_dvt[tid] = error_signal[tid] * act_dvt[tid];
  }
}

__global__ void Update_Param_Gpu(float * x, float * dx, float ALPHA, int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < n)
  {
    // x = x + alpha * dx
    x[tid] = x[tid] + ALPHA * dx[tid];

    // clear dx
    dx[tid] = 0;
  }
}

__global__ void Softmax_Gpu(const float * z, float * output, const int n)
{
  float sum = 0.0;
  int   tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < n)
  {
    output[tid] = exp(z[tid]);
  }

  __syncthreads();

  if(tid < n)
  {
    for(int i = 0; i < n; i++)
    {
      sum += output[i];
    }
  }

  __syncthreads();

  if(tid < n)
  {
    output[tid] /= sum;
  }
}

__global__ void Softmax_Dev_Gpu(const float * output, float * act_dvt, const int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < n)
  {
    act_dvt[tid] = output[tid] * (1 - output[tid]);
  }
}

__global__ void fill_rand_gpu(float * array, int seed, int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < n)
  {
    curandState_t state;
    curand_init(seed, tid, 0, &state);

    // create random number in range [-0.5 , 0.5] with uniform distribution
    array[tid] = curand_uniform(&state) - 0.5;
  }
}

__global__ void fill_zero_gpu(float * array, int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < n)
  {
    array[tid] = 0;
  }
}

/**************************************************************
 *
 *
 * HALF PRECISION FLOATING POINT KERNELS
 *
 *
 **************************************************************/
__global__ void h_add_vectors(const half* x, half* y, const int n)
{
 int tid = blockIdx.x * blockDim.x + threadIdx.x;

 if(tid  < n)
 {
   half alpha = __float2half(1.0);
   // y = alpha * x + y
   y[tid] = __hfma(alpha, x[tid], y[tid]);
 }
}

__global__ void cvt_float2half_gpu(const float * src, layer_param_t dst, const int n)
{
 int tid = blockIdx.x * blockDim.x + threadIdx.x;

 if(tid < n)
 {
   dst[tid] = __float2half(src[tid]);
 }
}

__global__ void cvt_half2float_gpu(const layer_param_t src, float * dst, const int n)
{
 int tid = blockIdx.x * blockDim.x + threadIdx.x;

 if(tid < n)
 {
   dst[tid] = __half2float(src[tid]);
 }
}

__global__ void h_fill_zero_gpu(half * array, int n)
{
 int tid = blockIdx.x * blockDim.x + threadIdx.x;

 if(tid < n)
 {
   array[tid] = __float2half(0);
 }
}

__global__ void h_fill_rand_gpu(half * array, int seed, int n)
{
 int tid = blockIdx.x * blockDim.x + threadIdx.x;

 if(tid < n)
 {
   curandState_t state;
   curand_init(seed, tid, 0, &state);

   // create random number in range [-0.5 , 0.5] with uniform distribution
   array[tid] = __float2half( curand_uniform(&state) - 0.5 );
 }
}

__global__ void h_Softmax_Dev_Gpu(const half * output, half * act_dvt, const int n)
{
 int tid = blockIdx.x * blockDim.x + threadIdx.x;

 if(tid < n)
 {
   // act_dvt[tid] = output[tid] * (1 - output[tid]);
   half one       = __float2half(1.0);
   half minus_one = __float2half(-1.0);
   half temp      = __hfma(minus_one, output[tid], one);
   act_dvt[tid]   = __hmul(output[tid], temp);

 }
}


__global__ void h_Softmax_Gpu(const half * z, half * output, const int n)
{
 half sum = __float2half(0.0);
 int   tid = blockIdx.x * blockDim.x + threadIdx.x;

 if(tid < n)
 {
   // output[tid] = exp(z[tid]);
   output[tid] = hexp(z[tid]);
 }

 __syncthreads();

 if(tid < n)
 {
   for(int i = 0; i < n; i++)
   {
     // sum += output[i];
     sum = __hadd( sum , output[i] );
   }
 }

 __syncthreads();

 if(tid < n)
 {
   // output[tid] /= sum;
   output[tid] = __hdiv(output[tid], sum);
 }
}

__global__ void h_Update_Param_Gpu(half * x, half * dx, float ALPHA, const int n)
{
 int tid = blockIdx.x * blockDim.x + threadIdx.x;

 if(tid < n)
 {
   half h_alpha = __float2half(ALPHA);

   // x = x + alpha * dx
   // x[tid] = x[tid] + ALPHA * dx[tid];
   x[tid] = __hfma(h_alpha, dx[tid], x[tid]);

   // clear dx
   // dx[tid] = 0;
   dx[tid] = __float2half(0.0);

 }
}

__global__ void h_Err_Dev_Gpu(const half * error_signal, const half * act_dvt, half * err_dvt, const int n)
{
 int tid = blockIdx.x * blockDim.x + threadIdx.x;

 if(tid < n)
 {
   // err_dvt[tid] = error_signal[tid] * act_dvt[tid];
   err_dvt[tid] = __hmul(error_signal[tid], act_dvt[tid]);
 }
}

__global__ void h_Sigmoid_Dev_Gpu(const half * output, half * act_dvt, const int n)
{
 int tid = blockIdx.x * blockDim.x + threadIdx.x;

 if(tid < n)
 {
   // act_dvt[tid] = output[tid] * (1 - output[tid]);
   half one       = __float2half(1.0);
   half minus_one = __float2half(-1.0);
   half temp      = __hfma(minus_one, output[tid], one);
   act_dvt[tid]   = __hmul(output[tid], temp);

 }
}

__global__ void h_Sigmoid_Gpu(const half * z, half * output, const int n)
{
 int tid = blockIdx.x * blockDim.x + threadIdx.x;

 if(tid < n)
 {
   // output[tid] = 1.0 / (1.0 + exp(-z[tid]));
   half exponent = hexp(z[tid]);
   half num      = __float2half(1.0);
   half divisor  = __hfma(num, exponent, num);

   output[tid]   = __hdiv(exponent, divisor);

 }
}

__global__ void h_CrossEntropyLoss_Derivative_Gpu(const half * neural_out, const half * expect_out, half * loss_dvt, const int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < n)
  {
   //  loss_dvt[tid] = -expect_out[tid] / neural_out[tid] + (1 - expect_out[tid]) / (1 - neural_out[tid]);
   half minus_one = __float2half(-1.0);
   half one       = __float2half(1.0);
   half x         = __hdiv(expect_out[tid], neural_out[tid]);
   half y         = __hdiv( __hfma(minus_one, expect_out[tid], one), __hfma(minus_one, neural_out[tid], one) );
   loss_dvt[tid]  = __hfma(minus_one, x, y);

  }
}

/*************************************************************
 *    PUBLIC FUNCTIONS
 *************************************************************/
/***************************************
 *  FLOAT - HALF FLOAT CONVERT FUNCTION
 ***************************************/
void Helper::cvtfloat2half(const float * src, Layer::layer_param_t dst, const int n_elements)
{
  cvt_float2half_gpu<<<CUDA_BLOCKS(size), Device::total_threads>>>(src, dst, n_elements);
}

void Helper::cvthalf2float(const Layer::layer_param_t src, float * dst, const int n_elements)
{
  cvt_half2float_gpu<<<CUDA_BLOCKS(size), Device::total_threads>>>(src, dst, n_elements);
}

/***************************************
 *  MEMORY ALLOCATION FUNCTION
 ***************************************/
void Helper::cuda_array_random_allocate(void **array, Layer::param_type_e type, int size)
{
  if( type == Layer::FLOAT_TYPE )
  {
    cudaMalloc(array, size * sizeof(float));
    // Fill with random number
    fill_rand_gpu<<<CUDA_BLOCKS(size), Device::total_threads>>>(*array, time(NULL), size);
  }
  else if( type == Layer::HALF_FLOAT_TYPE )
  {
    cudaMalloc(array, size * sizeof(half));
    h_fill_rand_gpu<<<CUDA_BLOCKS(size), Device::total_threads>>>(*array, time(NULL), size);
  }

}

void Helper::cuda_array_zero_allocate(void **array, Layer::param_type_e type, int size)
{
  if( type == Layer::FLOAT_TYPE )
  {
    cudaMalloc(array, size * sizeof(float));
    // fill with zero
    fill_zero_gpu<<<CUDA_BLOCKS(size), Device::total_threads>>>(*array, size);
  }
  else if( type == Layer::HALF_FLOAT_TYPE )
  {
    cudaMalloc(array, size * sizeof(half));
    h_fill_zero_gpu<<<CUDA_BLOCKS(size), Device::total_threads>>>(*array, size);
  }

}

void Helper::cuda_array_allocate(void **array, Layer::param_type_e type, int size)
{
  if( type == Layer::FLOAT_TYPE )
  {
    cudaMalloc(array, size * sizeof(float));
  }
  else if( type == Layer::HALF_FLOAT_TYPE )
  {
    cudaMalloc(array, size * sizeof(half));
  }
}

/***************************************
 *  LAYER SUB-CALCULATION FUNCTION
 ***************************************/
 void Helper::net_calc(const Layer::layer_param_t input, const Layer::layer_param_t w,
                              const Layer::layer_param_t b, Layer::layer_param_t z,
                              int total_inputs, int total_outputs)
 {
   if( Helper::network_type_get() == Layer::FLOAT_TYPE )
   {
      float alpha = 1.0;
      float beta  = 0.0;

      int m = 1;              // number of rows of matrix op(A) and C
      int n = total_outputs;  // number of columns of matrix op (B) and C
      int k = total_inputs;   // number of columns and rows of matrix op(A) and op(B)

      int lda = 1;            // leading dimension of matrix A
      int ldb = total_inputs; // leading dimension of matrix B
      int ldc = 1;            // leading dimension of matrix C

      float *mat_a = (float *)input;  // Matrix A
      float *mat_b = (float *)w;      // Matrix B
      float *mat_c = z;               // Matrix C

      cublasOperation_t op_A = CUBLAS_OP_N; // op(A) = A
      cublasOperation_t op_B = CUBLAS_OP_N; // op(B) = B

      // calculate z = x*W
      cublasSgemm(Device::Device_Get_Handle(),op_A,op_B,\
                      m , n , k,\
                      &alpha,\
                      mat_a , lda,\
                      mat_b , ldb,\
                      &beta ,\
                      mat_c , ldc);

      // add bias z = bias + z
      cublasSaxpy(Device::Device_Get_Handle(), n, &alpha, b, 1, z, 1);
   }
   else if( Helper::network_type_get() == Layer::HALF_FLOAT_TYPE )
   {
      half alpha = approx_float_to_half(1.0);
      half beta  = approx_float_to_half(0.0);

      int m = 1;              // number of rows of matrix op(A) and C
      int n = total_outputs;  // number of columns of matrix op (B) and C
      int k = total_inputs;   // number of columns and rows of matrix op(A) and op(B)

      int lda = 1;            // leading dimension of matrix A
      int ldb = total_inputs; // leading dimension of matrix B
      int ldc = 1;            // leading dimension of matrix C

      half *mat_a = input;    // Matrix A
      half *mat_b = w;        // Matrix B
      half *mat_c = z;        // Matrix C

      cublasOperation_t op_A = CUBLAS_OP_N; // op(A) = A
      cublasOperation_t op_B = CUBLAS_OP_N; // op(B) = B

      // calculate z = x*W
      cublasHgemm(Device::Device_Get_Handle(),op_A,op_B,\
                   m , n , k,\
                   &alpha,\
                   mat_a , lda,\
                   mat_b , ldb,\
                   &beta ,\
                   mat_c , ldc);

      // add bias z = bias + z
      h_add_vectors<<<CUDA_BLOCKS(total_outputs), Device::total_threads>>>(b, z, total_outputs);
   }
 }

 void Helper::sigmoid_calc(const Layer::layer_param_t z, Layer::layer_param_t output, int n)
 {
   if( Helper::network_type_get() == Layer::FLOAT_TYPE )
   {
     Sigmoid_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(z, output, n);
   }
   else if( Helper::network_type_get() == Layer::HALF_FLOAT_TYPE )
   {
     h_Sigmoid_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(z, output, n);
   }
 }

 void Helper::sigmoid_dev_calc(Layer::layer_param_t output, Layer::layer_param_t act_dvt, int n)
 {
   if( Helper::network_type_get() == Layer::FLOAT_TYPE )
   {
     Sigmoid_Dev_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(output, act_dvt, n);
   }
   else if( Helper::network_type_get() == Layer::HALF_FLOAT_TYPE )
   {
     h_Sigmoid_Dev_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(output, act_dvt, n);
   }
 }

void Helper::softmax_calc(const Layer::layer_param_t z, Layer::layer_param_t output, int n)
{
  if( Helper::network_type_get() == Layer::FLOAT_TYPE )
  {
    Softmax_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(z, output, n);
  }
  else if( Helper::network_type_get() == Layer::HALF_FLOAT_TYPE )
  {
    h_Softmax_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(z, output, n);
  }

}

void Helper::softmax_dev_calc(const Layer::layer_param_t output, Layer::layer_param_t act_dvt, int n)
{
  if( Helper::network_type_get() == Layer::FLOAT_TYPE )
  {
    Softmax_Dev_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(output, act_dvt, n);
  }
  else if( Helper::network_type_get() == Layer::HALF_FLOAT_TYPE )
  {
    h_Softmax_Dev_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(output, act_dvt, n);
  }

}

void Helper::err_dev_calc(Layer::layer_param_t error_signal, Layer::layer_param_t act_dvt,
                            Layer::layer_param_t err_dvt, int n)
{
  if( Helper::network_type_get() == Layer::FLOAT_TYPE )
  {
    Err_Dev_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(error_signal, act_dvt, err_dvt, n);
  }
  else if( Helper::network_type_get() == Layer::HALF_FLOAT_TYPE )
  {
    h_Err_Dev_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(error_signal, act_dvt, err_dvt, n);
  }

}

void Helper::accum_w_grad(Layer::layer_param_t input, Layer::layer_param_t err_dvt,
                            Layer::layer_param_t w_grad, int total_inputs, int total_outputs)
{
  if( Helper::network_type_get() == Layer::FLOAT_TYPE )
  {
    float alpha = 1.0;
    float beta  = 1.0;

    int m = total_inputs;   // number of rows of matrix op(A) and C
    int n = total_outputs;  // number of columns of matrix op (B) and C
    int k = 1;              // number of columns and rows of matrix op(A) and op(B)

    int lda = 1; // leading dimension of matrix A
    int ldb = 1; // leading dimension of matrix B
    int ldc = m; // leading dimension of matrix C

    float *mat_a = input;       // Matrix A
    float *mat_b = err_dvt;     // Matrix B
    float *mat_c = w_grad;      // Matrix C

    cublasOperation_t op_A = CUBLAS_OP_T; // op(A) = A'
    cublasOperation_t op_B = CUBLAS_OP_N; // op(B) = B

    // calculate C = alpha * A * B + beta * C
    // the formula is dW = dW + transpose(input) * err_dvt
    cublasSgemm(Device::Device_Get_Handle(),op_A,op_B,\
                    m     , n,  k,\
                    &alpha,\
                    mat_a , lda,\
                    mat_b , ldb,\
                    &beta ,\
                    mat_c , ldc);
  }
  else if( Helper::network_type_get() == Layer::HALF_FLOAT_TYPE )
  {
     half alpha = approx_float_to_half(1.0);
     half beta  = approx_float_to_half(1.0);

     int m = total_inputs;   // number of rows of matrix op(A) and C
     int n = total_outputs;  // number of columns of matrix op (B) and C
     int k = 1;              // number of columns and rows of matrix op(A) and op(B)

     int lda = 1; // leading dimension of matrix A
     int ldb = 1; // leading dimension of matrix B
     int ldc = m; // leading dimension of matrix C

     half *mat_a = input;       // Matrix A
     half *mat_b = err_dvt;     // Matrix B
     half *mat_c = w_grad;      // Matrix C

     cublasOperation_t op_A = CUBLAS_OP_T; // op(A) = A'
     cublasOperation_t op_B = CUBLAS_OP_N; // op(B) = B

     // calculate C = alpha * A * B + beta * C
     // the formula is dW = dW + transpose(input) * err_dvt
     cublasHgemm(Device::Device_Get_Handle(),op_A,op_B,\
                     m     , n,  k,\
                     &alpha,\
                     mat_a , lda,\
                     mat_b , ldb,\
                     &beta ,\
                     mat_c , ldc);

  }
}

void Helper::accum_b_grad(Layer::layer_param_t err_dvt, Layer::layer_param_t b_grad, int n)
{
  if( Helper::network_type_get() == Layer::FLOAT_TYPE )
  {
    float alpha = 1.0;
    float * x   = err_dvt;
    float * y   = b_grad;

    cublasSaxpy(Device::Device_Get_Handle(), n, &alpha, x, 1, y, 1);
  }
  else if( Helper::network_type_get() == Layer::HALF_FLOAT_TYPE )
  {
    half * x = err_dvt;
    half * y = b_grad;

    h_add_vectors<<<CUDA_BLOCKS(n), Device::total_threads>>>(x, y, n);
  }
}

void Helper::err_signal_calc(const Layer::layer_param_t w, const Layer::layer_param_t err_dvt,
                              Layer::layer_param_t propagate_err, int total_inputs, int total_outputs)
{
  if( Helper::network_type_get() == Layer::FLOAT_TYPE )
  {
    float alpha = 1.0;
    float beta  = 0.0;

    int m = 1;              // number of rows of matrix op(A) and C
    int n = total_inputs;   // number of columns of matrix op (B) and C
    int k = total_outputs;  // number of columns and rows of matrix op(A) and op(B)

    int lda = 1;            // leading dimension of matrix A
    int ldb = total_inputs; // leading dimension of matrix B
    int ldc = 1;            // leading dimension of matrix C

    float *mat_a = err_dvt;   // Matrix A
    float *mat_b = w;         // Matrix B
    float *mat_c = propagate_err;      // Matrix C

    cublasOperation_t op_A = CUBLAS_OP_N; // op(A) = A
    cublasOperation_t op_B = CUBLAS_OP_T; // op(B) = B'

    // calculate C = alpha * A * B + beta * C
    // the formula is pre_err = err_dvt * transpose(W)
    cublasSgemm(Device::Device_Get_Handle(),op_A,op_B,\
                    m     , n,  k,\
                    &alpha,\
                    mat_a , lda,\
                    mat_b , ldb,\
                    &beta ,\
                    mat_c , ldc);
  }
  else if( Helper::network_type_get() == Layer::HALF_FLOAT_TYPE )
  {
    half alpha = approx_float_to_half(1.0);
    half beta  = approx_float_to_half(0.0);

    int m = 1;              // number of rows of matrix op(A) and C
    int n = total_inputs;   // number of columns of matrix op (B) and C
    int k = total_outputs;  // number of columns and rows of matrix op(A) and op(B)

    int lda = 1;            // leading dimension of matrix A
    int ldb = total_inputs; // leading dimension of matrix B
    int ldc = 1;            // leading dimension of matrix C

    half *mat_a = err_dvt;   // Matrix A
    half *mat_b = w;         // Matrix B
    half *mat_c = propagate_err;      // Matrix C

    cublasOperation_t op_A = CUBLAS_OP_N; // op(A) = A
    cublasOperation_t op_B = CUBLAS_OP_T; // op(B) = B'

    // calculate C = alpha * A * B + beta * C
    // the formula is pre_err = err_dvt * transpose(W)
    cublasHgemm(Device::Device_Get_Handle(),op_A,op_B,\
                    m     , n,  k,\
                    &alpha,\
                    mat_a , lda,\
                    mat_b , ldb,\
                    &beta ,\
                    mat_c , ldc);
  }

}

void Helper::update_param(Layer::layer_param_t x, Layer::layer_param_t dx, float ALPHA, int n)
{
  if( Helper::network_type_get() == Layer::FLOAT_TYPE )
  {
    Update_Param_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(x, dx, ALPHA, n);
  }
  else if( Helper::network_type_get() == Layer::HALF_FLOAT_TYPE )
  {
    h_Update_Param_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(x, dx, ALPHA, n);
  }

}

/***************************************
 *  LOSS FUNCTION
 ***************************************/
void Helper::Cross_Entropy_Loss(const float * neural_out, const float * expect_out, float * loss, int n)
{
  float sum = 0.0;
  for(int i = 0; i < n; i++)
  {
    sum += -( expect_out[i] * log(neural_out[i]) + (1 - expect_out[i]) * log(1 - neural_out[i]) );
  }

  *loss = sum / n;
}

void Helper::Cross_Entropy_Loss_Derivative(const Layer::layer_param_t neural_out, const Layer::layer_param_t expect_out,
                                            Layer::layer_param_t loss_dvt, int n)
{
  if( Helper::network_type_get() == Layer::FLOAT_TYPE )
  {
    CrossEntropyLoss_Derivative_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(neural_out, expect_out, loss_dvt, n);
  }
  else if( Helper::network_type_get() == Layer::HALF_FLOAT_TYPE )
  {
    h_CrossEntropyLoss_Derivative_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(neural_out, expect_out, loss_dvt, n);
  }

}

Layer::param_type_e   Helper::network_type_get(void)
{
#if USING_HALF_FLOAT
  return Layer::HALF_FLOAT_TYPE;
#else
  return Layer::FLOAT_TYPE;
#endif
}
