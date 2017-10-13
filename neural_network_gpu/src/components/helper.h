#ifndef __HELPER_H
#define __HELPER_H

#include "../layers/layer.h"

class Helper
{
public:
  static Layer::param_type_e   network_type_get(void);
  /************* FLOAT - HALF FLOAT CONVERT FUNCTION *************/
  static void cvtfloat2half(const float * src, Layer::layer_param_t dst, const int n_elements);
  static void cvthalf2float(const Layer::layer_param_t src, float * dst, const int n_elements);

  /************* MEMORY ALLOCATION FUNCTION *************/
  static void cuda_array_random_allocate(void **array, Layer::param_type_e type, int size);
  static void cuda_array_zero_allocate(void **array, Layer::param_type_e type, int size);
  static void cuda_array_allocate(void **array, Layer::param_type_e type, int size);

  /************* LAYER SUB-CALCULATION FUNCTION *************/
  static void net_calc(const Layer::layer_param_t input, const Layer::layer_param_t w,
                          const Layer::layer_param_t b, Layer::layer_param_t z,
                          int total_inputs, int total_outputs);
  static void sigmoid_calc(const Layer::layer_param_t z, Layer::layer_param_t output, int n);
  static void sigmoid_dev_calc(Layer::layer_param_t output, Layer::layer_param_t act_dvt, int n);

  static void softmax_calc(const Layer::layer_param_t z, Layer::layer_param_t output, int n);
  static void softmax_dev_calc(const Layer::layer_param_t output, Layer::layer_param_t act_dvt, int n);

  static void err_dev_calc(Layer::layer_param_t error_signal, Layer::layer_param_t act_dvt, Layer::layer_param_t err_dvt, int n);
  static void accum_w_grad(Layer::layer_param_t input, Layer::layer_param_t err_dvt,
                            Layer::layer_param_t w_grad,  int total_inputs, int total_outputs);
  static void accum_b_grad(Layer::layer_param_t err_dvt, Layer::layer_param_t b_grad, int n);
  static void err_signal_calc(const Layer::layer_param_t w, const Layer::layer_param_t err_dvt,
                                Layer::layer_param_t propagate_err, int total_inputs, int total_outputs);
  static void update_param(Layer::layer_param_t x, Layer::layer_param_t dx, float ALPHA, int n);

  /************* LOSS FUNCTION *************/
  static void Cross_Entropy_Loss(const float * neural_out, const float * expect_out, float * loss, int n);
  static void Cross_Entropy_Loss_Derivative(const Layer::layer_param_t neural_out, const Layer::layer_param_t expect_out,
                                              Layer::layer_param_t loss_dvt, int n);
};







#endif
