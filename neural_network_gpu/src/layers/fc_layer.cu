/*************************************************************
*   File: fc_layer.cu
*
*
*************************************************************/
#include <iostream>
#include <cmath>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "fc_layer.h"
#include "../components/device.h"
#include "../components/helper.h"
/*************************************************************
 *    PUBLIC FUNCTIONS
 *************************************************************/
FC_Layer::FC_Layer(int n_inputs, int n_outputs)
{
  total_inputs  = n_inputs;
  total_outputs = n_outputs;

#if USING_HALF_FLOAT
  Layer::param_type_e type = Layer::HALF_FLOAT_TYPE
#else
  Layer::param_type_e type = Layer::FLOAT_TYPE;
#endif

  Helper::cuda_array_random_allocate( (void **)&w    , type, n_inputs * n_outputs );
  Helper::cuda_array_random_allocate( (void **)&b    , type, n_outputs );

  Helper::cuda_array_zero_allocate( (void **)&z      , type, n_outputs );
  Helper::cuda_array_zero_allocate( (void **)&w_grad , type, n_inputs * n_outputs );
  Helper::cuda_array_zero_allocate( (void **)&b_grad , type, n_outputs );
  Helper::cuda_array_zero_allocate( (void **)&output , type, n_outputs );
  Helper::cuda_array_zero_allocate( (void **)&err    , type, n_inputs );
  Helper::cuda_array_zero_allocate( (void **)&act_dvt, type, n_outputs );
  Helper::cuda_array_zero_allocate( (void **)&err_dvt, type, n_outputs );
}


FC_Layer::~FC_Layer()
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

Layer::layer_param_t FC_Layer::forward_propagation(Layer::layer_param_t in)
{
  // Save the input
  input = in;

  // Calculate the net
  // z = x.w + b
  Helper::net_calc(input, w, b, z, total_inputs, total_outputs);

  // Apply Sigmoid activate function
  // output = sigmoid(z)
  Helper::sigmoid_calc(z, output, total_outputs);

  // Return this layer's output for further calculation in next layer
  return output;
}

Layer::layer_param_t FC_Layer::backward_propagation(Layer::layer_param_t error)
{
  // Calculate derivative of neuron output
  // dO/dnet = sigmoid'(z)
  Helper::sigmoid_dev_calc(output, act_dvt, total_outputs);

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

void FC_Layer::update(float eta, int batch_size)
{
  // Update weights and biases and clear gradients
  // w = w - dw * (eta/batch_size)
  // b = b - db * (eta/batch_size)
  float alpha = -eta / batch_size;
  Helper::update_param(w, w_grad, alpha, total_inputs * total_outputs);
  Helper::update_param(b, b_grad, alpha, total_outputs);


}
/*************************************************************
 *    PRIVATE FUNCTIONS
 *************************************************************/
