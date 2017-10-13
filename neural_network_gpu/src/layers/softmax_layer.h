#ifndef __SOFTMAX_LAYER_H
#define __SOFTMAX_LAYER_H

#include "layer.h"

class Softmax_Layer : public Layer
{
public:
  Softmax_Layer(int n_inputs, int n_outputs);
  virtual          ~Softmax_Layer();
  layer_param_t    forward_propagation(layer_param_t in);
  layer_param_t    backward_propagation(layer_param_t error);
  void             update(float eta, int batch_size);

};

#endif
