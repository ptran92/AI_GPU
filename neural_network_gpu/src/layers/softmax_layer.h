#ifndef __SOFTMAX_LAYER_H
#define __SOFTMAX_LAYER_H

#include "layer.h"

class Softmax_Layer : public Layer
{
public:
  Softmax_Layer(int n_inputs, int n_outputs);
  virtual          ~Softmax_Layer();
  Layer::layer_param_t    forward_propagation(Layer::layer_param_t in);
  Layer::layer_param_t    backward_propagation(Layer::layer_param_t error);
  void             update(float eta, int batch_size);

};

#endif
