#ifndef __LAYER_H
#define __LAYER_H

class Layer
{
public:
  virtual float *    forward_propagation(float * in)     = 0;
  virtual float *    backward_propagation(float * error) = 0;
  virtual void       update(float eta, int batch_size)   = 0;
  virtual void       test(void) = 0;

protected:
  int     total_inputs;
  int     total_outputs;
  float * input;
  float * w;
  float * b;
  float * z;
  float * w_grad;
  float * b_grad;
  float * output;
  float * err;
  float * act_dvt;
  float * err_dvt;
};


#endif
