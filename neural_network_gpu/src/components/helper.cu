/*************************************************************
*   File: helper.cu
*
*
*************************************************************/
#include <ctime>
#include <random>
#include <memory>
#include "helper.h"
#include "cublas_v2.h"
#include "curand.h"
#include "curand_kernel.h"
#include "cuda_runtime.h"
#include "device.h"
#include "fp16_conversion.h"
/*************************************************************
 *    CONSTANTS
 *************************************************************/
#define MAX_HALF_FLOAT		(65500.0)
#define MIN_HALF_FLOAT		(-65500.0)

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
__global__ void cvt_float2half_gpu(const float * src, half * dst, const int n)
{
 int tid = blockIdx.x * blockDim.x + threadIdx.x;

 if(tid < n)
 {
   dst[tid] = __float2half(src[tid]);
 }
}

__global__ void cvt_half2float_gpu(const half * src, float * dst, const int n)
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

   array[tid] = __float2half( curand_normal(&state)*0.1 ); // create normal distributed random numbers, mean 0, standard deviation 0.1
 }
}

#if USING_HALF_FLOAT

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
 int  tid = blockIdx.x * blockDim.x + threadIdx.x;
 half max = z[0];

 /* find the max in input buffer */
 if(tid < n)
 {
   for(int i = 1; i < n; i++)
   {
     if( __hgt(z[i], max) )
     {
       max = z[i];
     }
   }
 }

 __syncthreads();

 if(tid < n)
 {
   // output[tid] = exp( z[tid] - max );
   output[tid] = hexp( __hsub(z[tid], max) );
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
   output[tid] = hdiv(output[tid], sum);

   // trick! add or substract an epsilon to prevent output to zero or one
   half zero = __float2half(0.0);
   half one  = __float2half(1.0);
   half esp  = __float2half(0.001);
   if( __heq(output[tid], zero) )
   {
     // if equal to zero, add an epsilon
     output[tid] = __hadd(output[tid], esp);
   }
   else if( __heq(output[tid], one)  )
   {
     // if equal to one, substract an epsilon
     output[tid] = __hsub(output[tid], esp);
   }

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
   /*
     output[tid] = 1.0 / (1.0 + exp(-z[tid]));

     NOTE: for 16-bit floating point number, to solve the overflow and underflow problem
     in sigmoid function, a solution is proposed:
     + If z > 0:
        sigmoid(z) = 1 / (1 + e^-z)
     + If z <= 0:
        sigmoid(z) = e^z / (1 + e^z)
   */
   half one       = __float2half(1.0);
   half minus_one = __float2half(-1.0);
   half zero      = __float2half(0.0);

   if( __hgt(z[tid], zero) )
   {
     /*
       if z is greater than 0
       use the formula sigmoid(x) = 1 / (1 + e^-x)
     */
     half temp      = __hfma(minus_one, z[tid], zero);
     half divisor   = __hfma(one, hexp(temp), one);
     output[tid]    = hdiv(one, divisor);

   }
   else
   {
     /*
       if z is equal or less than 0
       use the formula sigmoid(x) = e^x / (1 + e^x)
     */
     half exponent  = hexp(z[tid]);
     half divisor   = __hfma(one, exponent, one);
     output[tid]    = hdiv(exponent, divisor);
   }

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
   half y         = neural_out[tid];

   half a         = hdiv(expect_out[tid], y);
   half b         = hdiv( __hfma(minus_one, expect_out[tid], one), __hfma(minus_one, y, one) );

   loss_dvt[tid]  = __hfma(minus_one, a, b);

   /* if result is inf, replace it with maximum value of float 16 */
   int is_inf     = __hisinf(loss_dvt[tid]);
   if( is_inf == -1 )
   {
    // if negative infinity
    loss_dvt[tid] = __float2half( MIN_HALF_FLOAT );
   }
   else if ( is_inf == 1 )
   {
     // if positive infinity
    loss_dvt[tid] = __float2half( MAX_HALF_FLOAT );
   }

  }
}


__global__ void h_Self_MultiplyMatrix(const half* x, const half* y, half* z, int row_x, int col_x, int col_y)
{
  int tid   = blockIdx.x * blockDim.x + threadIdx.x;
  int n2    = col_y/2;

  if(tid < n2)
  {
      half2 *output = (half2 *)z;
      half2 sum     = __float2half2_rn(0.0f);
      int count     = col_x;
      int l_col     = count * 2 * tid;
      int r_col     = count * (2 * tid + 1);

      for(int i = 0; i < count; i++)
      {
        sum = __hfma2(__half2half2(x[i]), __halves2half2(y[i + l_col], y[i + r_col]), sum);
      }

      output[tid] = sum;
  }
}

__global__ void h_AddVector(half * x, const half * y, int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n2 = n/2;

  if(tid < n2)
  {
    half2 *a = (half2 *)x;
    half2 *b = (half2 *)y;

    a[tid] = __hadd2(a[tid], b[tid]);

  }
}


#endif

/*************************************************************
 *    PUBLIC FUNCTIONS
 *************************************************************/
/***************************************
 *  FLOAT - HALF FLOAT CONVERT FUNCTION
 ***************************************/
void Helper::cvtfloat2half(const float * src, Layer::layer_param_t dst, const int n_elements)
{
#if USING_HALF_FLOAT
  cvt_float2half_gpu<<<CUDA_BLOCKS(n_elements), Device::total_threads>>>(src, dst, n_elements);
#endif
}

void Helper::cvthalf2float(const Layer::layer_param_t src, float * dst, const int n_elements)
{
#if USING_HALF_FLOAT
  cvt_half2float_gpu<<<CUDA_BLOCKS(n_elements), Device::total_threads>>>(src, dst, n_elements);
#endif
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
    fill_rand_gpu<<<CUDA_BLOCKS(size), Device::total_threads>>>((float *)*array, time(NULL), size);
  }
  else if( type == Layer::HALF_FLOAT_TYPE )
  {
    cudaMalloc(array, size * sizeof(half));
    h_fill_rand_gpu<<<CUDA_BLOCKS(size), Device::total_threads>>>((half *)*array, time(NULL), size);
  }

}

void Helper::cuda_array_zero_allocate(void **array, Layer::param_type_e type, int size)
{
  if( type == Layer::FLOAT_TYPE )
  {
    cudaMalloc(array, size * sizeof(float));
    // fill with zero
    fill_zero_gpu<<<CUDA_BLOCKS(size), Device::total_threads>>>((float *)*array, size);
  }
  else if( type == Layer::HALF_FLOAT_TYPE )
  {
    cudaMalloc(array, size * sizeof(half));
    h_fill_zero_gpu<<<CUDA_BLOCKS(size), Device::total_threads>>>((half *)*array, size);
  }

}

void Helper::cuda_array_allocate(void **array, Layer::param_type_e type, int size)
{
  if( type == Layer::FLOAT_TYPE )
  {;
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
 #if USING_HALF_FLOAT
    // half alpha = approx_float_to_half(1.0);
    // half beta  = approx_float_to_half(0.0);
    //
    // int m = 1;              // number of rows of matrix op(A) and C
    // int n = total_outputs;  // number of columns of matrix op (B) and C
    // int k = total_inputs;   // number of columns and rows of matrix op(A) and op(B)
    //
    // int lda = 1;            // leading dimension of matrix A
    // int ldb = total_inputs; // leading dimension of matrix B
    // int ldc = 1;            // leading dimension of matrix C
    //
    // half *mat_a = input;    // Matrix A
    // half *mat_b = w;        // Matrix B
    // half *mat_c = z;        // Matrix C
    //
    // cublasOperation_t op_A = CUBLAS_OP_N; // op(A) = A
    // cublasOperation_t op_B = CUBLAS_OP_N; // op(B) = B
    //
    // // calculate z = x*W
    // cublasHgemm(Device::Device_Get_Handle(),op_A,op_B,\
    //             m , n , k,\
    //             &alpha,\
    //             mat_a , lda,\
    //             mat_b , ldb,\
    //             &beta ,\
    //             mat_c , ldc);
    //
    // // add bias z = bias + z
    // h_add_vectors<<<CUDA_BLOCKS(total_outputs), Device::total_threads>>>(b, z, total_outputs);

    /* Z = X * W */
    h_Self_MultiplyMatrix<<<CUDA_BLOCKS(total_outputs), Device::total_threads>>>(input, w, z, 1, total_inputs, total_outputs);

    /* Z = Z + B */
    h_AddVector<<<CUDA_BLOCKS(total_outputs), Device::total_threads>>>(z, b, total_outputs);

 #else
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

 #endif

 }

 void Helper::sigmoid_calc(const Layer::layer_param_t z, Layer::layer_param_t output, int n)
 {
 #if USING_HALF_FLOAT
    h_Sigmoid_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(z, output, n);
#else
    Sigmoid_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(z, output, n);
 #endif

 }

 void Helper::sigmoid_dev_calc(Layer::layer_param_t output, Layer::layer_param_t act_dvt, int n)
 {
 #if USING_HALF_FLOAT
    h_Sigmoid_Dev_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(output, act_dvt, n);
 #else
    Sigmoid_Dev_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(output, act_dvt, n);
 #endif

 }

void Helper::softmax_calc(const Layer::layer_param_t z, Layer::layer_param_t output, int n)
{
#if USING_HALF_FLOAT
   h_Softmax_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(z, output, n);
#else
   Softmax_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(z, output, n);
#endif

}

void Helper::softmax_dev_calc(const Layer::layer_param_t output, Layer::layer_param_t act_dvt, int n)
{
#if USING_HALF_FLOAT
   h_Softmax_Dev_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(output, act_dvt, n);
#else
   Softmax_Dev_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(output, act_dvt, n);
#endif

}

void Helper::err_dev_calc(Layer::layer_param_t error_signal, Layer::layer_param_t act_dvt,
                            Layer::layer_param_t err_dvt, int n)
{
#if USING_HALF_FLOAT
   h_Err_Dev_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(error_signal, act_dvt, err_dvt, n);
#else
   Err_Dev_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(error_signal, act_dvt, err_dvt, n);
#endif

}

void Helper::accum_w_grad(Layer::layer_param_t input, Layer::layer_param_t err_dvt,
                            Layer::layer_param_t w_grad, int total_inputs, int total_outputs)
{
#if USING_HALF_FLOAT
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

#else
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

#endif

}

void Helper::accum_b_grad(Layer::layer_param_t err_dvt, Layer::layer_param_t b_grad, int n)
{
#if USING_HALF_FLOAT
  half * x = err_dvt;
  half * y = b_grad;

  h_add_vectors<<<CUDA_BLOCKS(n), Device::total_threads>>>(x, y, n);
#else
  float alpha = 1.0;
  float * x   = err_dvt;
  float * y   = b_grad;

  cublasSaxpy(Device::Device_Get_Handle(), n, &alpha, x, 1, y, 1);
#endif

}

void Helper::err_signal_calc(const Layer::layer_param_t w, const Layer::layer_param_t err_dvt,
                              Layer::layer_param_t propagate_err, int total_inputs, int total_outputs)
{
#if USING_HALF_FLOAT
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

#else
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
#endif

}

void Helper::update_param(Layer::layer_param_t x, Layer::layer_param_t dx, float ALPHA, int n)
{
#if USING_HALF_FLOAT
  h_Update_Param_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(x, dx, ALPHA, n);
#else
  Update_Param_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(x, dx, ALPHA, n);
#endif

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
#if USING_HALF_FLOAT
  h_CrossEntropyLoss_Derivative_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(neural_out, expect_out, loss_dvt, n);
#else
  CrossEntropyLoss_Derivative_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(neural_out, expect_out, loss_dvt, n);
#endif

}
/***************************************
 *  DEBUG FUNCTION
 ***************************************/
void Helper::Print_Array(const std::string buffer_name, const Layer::layer_param_t buffer, const int size)
{
#if 0
  /* Only print maximum 10 members or less */
  int no_elements_to_print = (size > 10)? 10 : size;

  /* Allocate temporary memory */
  std::unique_ptr<float []> cpu_buffer(new float[no_elements_to_print]);
  float *gpu_buffer;
  cudaMalloc(&gpu_buffer, no_elements_to_print * sizeof(float));

  /* Convert to single precision floating point buffer */
  Helper::cvthalf2float(buffer, gpu_buffer, no_elements_to_print);

  /* Copy to cpu memory */
  cudaMemcpy(cpu_buffer.get(), gpu_buffer, no_elements_to_print * sizeof(float), cudaMemcpyDeviceToHost);

  /* Print out the buffer */
  std::cout << buffer_name << ":" << std::endl;
  for(int i = 0; i < no_elements_to_print; i++)
  {
    std::cout << cpu_buffer[i] << " ";
  }
  std::cout << std::endl;

  cudaFree(gpu_buffer);
#endif
}
