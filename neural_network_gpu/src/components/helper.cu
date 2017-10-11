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
/*************************************************************
 *    KERNEL FUNCTIONS
 *************************************************************/
 __global__ void CrossEntropyLoss_Derivative_Gpu(const float * neural_out, const float * expect_out, float * loss_dvt, int n)
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
__global__ void Softmax_Gpu(const float * z, float * output, int n)
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

/*************************************************************
 *    PUBLIC FUNCTIONS
 *************************************************************/
void Helper::cuda_array_random_allocate(float **array, int size)
{
  cudaMalloc(array, size);

  // Fill with random number
  fill_rand_gpu<<<CUDA_BLOCKS(size), Device::total_threads>>>(*array, time(NULL), size);
  // => 96%
}

void Helper::cuda_array_zero_allocate(float **array, int size)
{
  cudaMalloc(array, size);

  // fill with zero
  fill_zero_gpu<<<CUDA_BLOCKS(size), Device::total_threads>>>(*array, size);
}

void Helper::cuda_array_allocate(float **array, int size)
{
  cudaMalloc(array, size);
}

 void Helper::net_calc(const float * input, const float * w, const float * b, float * z, int total_inputs, int total_outputs)
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

 void Helper::sigmoid_calc(const float * z, float * output, int n)
 {
   Sigmoid_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(z, output, n);
 }

 void Helper::sigmoid_dev_calc(float * output, float * act_dvt, int n)
 {
   Sigmoid_Dev_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(output, act_dvt, n);
 }

void Helper::softmax_calc(const float * z, float * output, int n)
{
  Softmax_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(z, output, n);
}

void Helper::softmax_dev_calc(const float * output, float * act_dvt, int n)
{
  Softmax_Dev_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(output, act_dvt, n);
}

void Helper::err_dev_calc(float * error_signal, float * act_dvt, float * err_dvt, int n)
{
  Err_Dev_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(error_signal, act_dvt, err_dvt, n);
}

void Helper::accum_w_grad(float * input, float * err_dvt, float * w_grad, int total_inputs, int total_outputs)
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

void Helper::accum_b_grad(float * err_dvt, float * b_grad, int n)
{
  float alpha = 1.0;
  float * x   = err_dvt;
  float * y   = b_grad;

  cublasSaxpy(Device::Device_Get_Handle(), n, &alpha, x, 1, y, 1);
}

void Helper::err_signal_calc(const float *w, const float * err_dvt, float *propagate_err, int total_inputs, int total_outputs)
{
  float alpha = 1.0;
  float beta  = 0.0;

  int m = 1;              // number of rows of matrix op(A) and C
  int n = total_inputs;   // number of columns of matrix op (B) and C
  int k = total_outputs;  // number of columns and rows of matrix op(A) and op(B)

  int lda = 1;            // leading dimension of matrix A
  int ldb = total_inputs; // leading dimension of matrix B
  int ldc = 1;            // leading dimension of matrix C

  float *mat_a = (float *)err_dvt;   // Matrix A
  float *mat_b = (float *)w;         // Matrix B
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

void Helper::update_param(float * x, float * dx, float ALPHA, int n)
{
  Update_Param_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(x, dx, ALPHA, n);
}

void Helper::Cross_Entropy_Loss(const float * neural_out, const float * expect_out, float * loss, int n)
{
  float sum = 0.0;
  for(int i = 0; i < n; i++)
  {
    sum += -( expect_out[i] * log(neural_out[i]) + (1 - expect_out[i]) * log(1 - neural_out[i]) );
  }

  *loss = sum / n;
}

void Helper::Cross_Entropy_Loss_Derivative(const float * neural_out, const float * expect_out, float * loss_dvt, int n)
{
  CrossEntropyLoss_Derivative_Gpu<<<CUDA_BLOCKS(n), Device::total_threads>>>(neural_out, expect_out, loss_dvt, n);
}
