#ifndef __HELPER_H
#define __HELPER_H

class Helper
{
public:
  static void cuda_array_random_allocate(float **array, int size);
  static void cuda_array_zero_allocate(float **array, int size);
  static void cuda_array_allocate(float **array, int size);

  static void net_calc(const float * input, const float * w, const float * b, float * z, int total_inputs, int total_outputs);
  static void sigmoid_calc(const float * z, float * output, int n);
  static void sigmoid_dev_calc(float * output, float * act_dvt, int n);

  static void softmax_calc(const float * z, float * output, int n);
  static void softmax_dev_calc(const float * output, float * act_dvt, int n);

  static void err_dev_calc(float * error_signal, float * act_dvt, float * err_dvt, int n);
  static void accum_w_grad(float * input, float * err_dvt, float * w_grad,  int total_inputs, int total_outputs);
  static void accum_b_grad(float * err_dvt, float * b_grad, int n);
  static void err_signal_calc(const float *w, const float * err_dvt, float *propagate_err, int total_inputs, int total_outputs);
  static void update_param(float * x, float * dx, float ALPHA, int n);

  /************* LOSS FUNCTION *************/
  static void Cross_Entropy_Loss_Derivative(const float * neural_out, const float * expect_out, float * loss_dvt, int n);
};







#endif
