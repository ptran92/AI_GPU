#ifndef __SOFTMAX_LAYER_H
#define __SOFTMAX_LAYER_H

#include "layer.h"

class Softmax_Layer : public Layer
{
public:
  Softmax_Layer(int n_inputs, int n_outputs);
  virtual ~Softmax_Layer();
  float *    forward_propagation(float * in);
  float *    backward_propagation(float * error);
  void       update(float eta, int batch_size);
  void       test();

};

#endif
