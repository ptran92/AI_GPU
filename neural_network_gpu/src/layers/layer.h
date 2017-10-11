#ifndef __LAYER_H
#define __LAYER_H

#define USING_HALF_FLOAT        1

class Layer
{
public:
  #if USING_HALF_FLOAT
    typedef short*     layer_param_t;
  #else
    typedef float*     layer_param_t;
  #endif

public:
  virtual layer_param_t    forward_propagation(layer_param_t in)     = 0;
  virtual layer_param_t    backward_propagation(layer_param_t error) = 0;
  virtual void       update(float eta, int batch_size)   = 0;
  virtual void       test(void) = 0;

protected:
  int     total_inputs;
  int     total_outputs;
  layer_param_t input;
  layer_param_t w;
  layer_param_t b;
  layer_param_t z;
  layer_param_t w_grad;
  layer_param_t b_grad;
  layer_param_t output;
  layer_param_t err;
  layer_param_t act_dvt;
  layer_param_t err_dvt;
};


#endif
