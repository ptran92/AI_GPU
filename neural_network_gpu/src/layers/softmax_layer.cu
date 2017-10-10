/*************************************************************
*   File: softmax_layer.cu
*
*
*************************************************************/
#include <iostream>
#include <cmath>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "softmax_layer.h"
#include "../components/device.h"
#include "../components/helper.h"
/*************************************************************
 *    PUBLIC FUNCTIONS
 *************************************************************/
Softmax_Layer::Softmax_Layer(int n_inputs, int n_outputs)
{
  total_inputs  = n_inputs;
  total_outputs = n_outputs;
  Helper::cuda_array_random_allocate( &w, sizeof(float) * n_inputs * n_outputs );
  Helper::cuda_array_random_allocate( &b, sizeof(float) * n_outputs );

  Helper::cuda_array_zero_allocate( &z        , sizeof(float) * n_outputs );
  Helper::cuda_array_zero_allocate( &w_grad   , sizeof(float) * n_inputs * n_outputs );
  Helper::cuda_array_zero_allocate( &b_grad   , sizeof(float) * n_outputs );
  Helper::cuda_array_zero_allocate( &output   , sizeof(float) * n_outputs );
  Helper::cuda_array_zero_allocate( &err      , sizeof(float) * n_inputs );
  Helper::cuda_array_zero_allocate( &act_dvt  , sizeof(float) * n_outputs );
  Helper::cuda_array_zero_allocate( &err_dvt  , sizeof(float) * n_outputs );

}

Softmax_Layer::~Softmax_Layer()
{
  cudaFree(w);
  cudaFree(b);
  cudaFree(z);
  cudaFree(w_grad);
  cudaFree(b_grad);
  cudaFree(output);
  cudaFree(err);
  cudaFree(act_dvt);
  cudaFree(err_dvt);
}

float * Softmax_Layer::forward_propagation(float * in)
{
  // Save the input
  input = in;

  // Calculate the net
  // z = w.x + b
  Helper::net_calc(input, w, b, z, total_inputs, total_outputs);

  // Apply Softmax activate function
  // output = softmax(z)
  Helper::softmax_calc(z, output, total_outputs);

  // Return this layer's output for further calculation in next layer
  return output;
}

float * Softmax_Layer::backward_propagation(float * error)
{
  // Calculate derivative of neuron output
  // dO/dnet = softmax'(z)
  Helper::softmax_dev_calc(output, act_dvt, total_outputs);

  // Calculate error derivative
  // dE/dnet = dE/dO x dO/dnet
  // dE/dO is error signal from next layer
  Helper::err_dev_calc(error, act_dvt, err_dvt, total_outputs);

  // Accumulate gradients
  // dw = dw + input.dE/dnet
  // db = db + dE/dnet
  Helper::accum_w_grad(input, err_dvt, w_grad, total_inputs, total_outputs);
  Helper::accum_b_grad(err_dvt, b_grad, total_outputs);

  // Calculate error signal propagated to previous layer
  // error_signal = dE/dnet * w
  Helper::err_signal_calc(w, err_dvt, err, total_inputs, total_outputs);

  // Back propagate this layer's error signal
  return err;
}

void Softmax_Layer::update(float eta, int batch_size)
{
  // Update weights and biases and clear gradients
  // w = w - dw * (eta/batch_size)
  // b = b - db * (eta/batch_size)
  float alpha = -eta / batch_size;
  Helper::update_param(w, w_grad, alpha, total_inputs * total_outputs);
  Helper::update_param(b, b_grad, alpha, total_outputs);
}

void Softmax_Layer::test(void)
{
  /*********************
   *  TEST EACH PRIVATE FUNCTION
   *********************/
  // Prepare inputs
  float * cpu_input         = new float[total_inputs];
  float * cpu_w             = new float[total_inputs * total_outputs];
  float * cpu_b             = new float[total_outputs];
  float * cpu_z             = new float[total_outputs];
  float * cpu_output        = new float[total_outputs];
  float * cpu_expect_output = new float[total_outputs];
  float * cpu_loss_dev      = new float[total_outputs];
  float * cpu_act_dvt       = new float[total_outputs];
  float * cpu_err_dvt       = new float[total_outputs];
  float * cpu_w_grad        = new float[total_inputs * total_outputs];
  float * cpu_b_grad        = new float[total_outputs];
  float * cpu_pre_err       = new float[total_inputs];

  for(int i = 0; i < total_inputs; i++)
    cpu_input[i] = i;

  for(int i = 0; i < total_outputs; i++)
  {
    cpu_expect_output[i] = i;
  }

  float * gpu_input;
  float * loss_dev;
  float * gpu_ex_output;
  cudaMalloc(&gpu_input     , sizeof(float) * total_inputs);
  cudaMalloc(&loss_dev      , sizeof(float) * total_outputs);
  cudaMalloc(&gpu_ex_output , sizeof(float) * total_outputs);

  cudaMemcpy(gpu_input    , cpu_input         , sizeof(float) * total_inputs  , cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_ex_output, cpu_expect_output , sizeof(float) * total_outputs , cudaMemcpyHostToDevice);


  cudaMemcpy(cpu_w, w, sizeof(float) * total_inputs * total_outputs, cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_b, b, sizeof(float) * total_outputs, cudaMemcpyDeviceToHost);

  /*********************
   *  BEGIN OF TEST
   *********************/
  /*
   *  net_calc
   */
  Helper::net_calc(gpu_input, w, b, z, total_inputs, total_outputs);
  cudaMemcpy(cpu_z, z, sizeof(float) * total_outputs, cudaMemcpyDeviceToHost);

  /*
   *  softmax_calc
   */
  Helper::softmax_calc(z, output, total_outputs);
  cudaMemcpy(cpu_output, output, sizeof(float) * total_outputs, cudaMemcpyDeviceToHost);

  /*
   * Calculate loss derivative
   */
  Helper::Cross_Entropy_Loss_Derivative(output, gpu_ex_output, loss_dev, total_outputs);
  cudaMemcpy(cpu_loss_dev, loss_dev, sizeof(float) * total_outputs, cudaMemcpyDeviceToHost);


  /*
   * Calculate derivative of neuron output
   */
  Helper::softmax_dev_calc(output, act_dvt, total_outputs);
  cudaMemcpy(cpu_act_dvt, act_dvt, sizeof(float) * total_outputs, cudaMemcpyDeviceToHost);

  /*
   * Calculate error derivative
   */
  Helper::err_dev_calc(loss_dev, act_dvt, err_dvt, total_outputs);
  cudaMemcpy(cpu_err_dvt, err_dvt, sizeof(float) * total_outputs, cudaMemcpyDeviceToHost);

  /*
   * Accumulate gradients
   */
  Helper::accum_w_grad(gpu_input, err_dvt, w_grad, total_inputs, total_outputs);
  Helper::accum_b_grad(err_dvt, b_grad, total_outputs);
  cudaMemcpy(cpu_w_grad, w_grad, sizeof(float) * total_inputs * total_outputs, cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_b_grad, b_grad, sizeof(float) * total_outputs, cudaMemcpyDeviceToHost);

  /*
   * Calculate error signal propagated to previous layer
   */
  Helper::err_signal_calc(w, err_dvt, err, total_inputs, total_outputs);
  cudaMemcpy(cpu_pre_err, err, sizeof(float) * total_inputs, cudaMemcpyDeviceToHost);

  /******************************************************************************
   * Display results
   ******************************************************************************/
  std::cout << "/*************** BEGIN OF TEST *****************/" << std::endl;
  std::cout << "Test file: " << __FILE__ << std::endl;
  std::cout << std::endl << "Input: " << std::endl;
  for(int i = 0; i < total_inputs; i++)
    std::cout << cpu_input[i] << " ";

  std::cout << std::endl << "W: " << std::endl;
  for(int i = 0; i < (total_inputs * total_outputs); i++)
    std::cout << cpu_w[i] << " ";

  std::cout << std::endl << "B: " << std::endl;
  for(int i = 0; i < (total_outputs); i++)
    std::cout << cpu_b[i] << " ";

  std::cout << std::endl << "Z: " << std::endl;
  for(int i = 0; i < (total_outputs); i++)
    std::cout << cpu_z[i] << " ";

  std::cout << std::endl << "OUTPUT: " << std::endl;
  for(int i = 0; i < (total_outputs); i++)
    std::cout << cpu_output[i] << " ";

  std::cout << std::endl << "EXPECT OUTPUT: " << std::endl;
  for(int i = 0; i < (total_outputs); i++)
    std::cout << cpu_expect_output[i] << " ";

  std::cout << std::endl << "LOSS DEV: " << std::endl;
  for(int i = 0; i < (total_outputs); i++)
    std::cout << cpu_loss_dev[i] << " ";

  std::cout << std::endl << "ACTIVATION DEV (SOFTMAX): " << std::endl;
  for(int i = 0; i < (total_outputs); i++)
    std::cout << cpu_act_dvt[i] << " ";

  std::cout << std::endl << "ERROR DEV: " << std::endl;
  for(int i = 0; i < (total_outputs); i++)
    std::cout << cpu_err_dvt[i] << " ";

  std::cout << std::endl << "W GRAD: " << std::endl;
  for(int i = 0; i < (total_inputs * total_outputs); i++)
    std::cout << cpu_w_grad[i] << " ";

  std::cout << std::endl << "B GRAD: " << std::endl;
  for(int i = 0; i < (total_outputs); i++)
    std::cout << cpu_b_grad[i] << " ";

  std::cout << std::endl << "PREVIOUS ERROR: " << std::endl;
  for(int i = 0; i < (total_inputs); i++)
    std::cout << cpu_pre_err[i] << " ";


  std::cout << std::endl << "/*************** END OF TEST *****************/" << std::endl;

  /*********************
   *  END OF TEST
   *********************/
  delete[] cpu_input;
  delete[] cpu_w;
  delete[] cpu_b;
  delete[] cpu_z;
  delete[] cpu_output;
  delete[] cpu_expect_output;
  delete[] cpu_loss_dev;
  delete[] cpu_act_dvt;
  delete[] cpu_err_dvt;
  delete[] cpu_w_grad;
  delete[] cpu_b_grad;
  delete[] cpu_pre_err;

  cudaFree(gpu_input);
  cudaFree(loss_dev);
  cudaFree(gpu_ex_output);

}

/*************************************************************
 *    PRIVATE FUNCTIONS
 *************************************************************/
