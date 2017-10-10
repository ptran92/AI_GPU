#ifndef __FC_LAYER_H
#define __FC_LAYER_H

#include "layer.h"

class FC_Layer : public Layer
{
public:
  FC_Layer(int n_inputs, int n_outputs);
  virtual ~FC_Layer();
  float *    forward_propagation(float * in);
  float *    backward_propagation(float * error);
  void       update(float eta, int batch_size);
  void       test();
};


#endif
