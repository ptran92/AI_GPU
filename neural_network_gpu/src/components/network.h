#ifndef __NETWORK_H
#define __NETWORK_H

#include <iostream>
#include <vector>
#include <memory>
#include "../layers/layer.h"

class Network
{
public:
  Network(std::vector<std::shared_ptr<Layer>>& group_l,
            int   input_size    ,
            int   output_size   ,
            float lr      = 0.01,
            int   b_size  = 10  ,
            int   epoch   = 10  );

  ~Network();

  void Predict(const float * input, float * output);

  void Train(const float * input, const float * e_output,  int total_train_samples,
              const float * test_input, const float * test_e_output, int total_test_samples);

private:
  Layer::layer_param_t Forward_Propagate(Layer::layer_param_t input);

private:
  std::vector<std::shared_ptr<Layer>>& layers;
  int       in_size;
  int       out_size;
  float     eta;
  int       batch_size;
  int       epoch_time;
  float *   gpu_input;
  float *   gpu_output;
  Layer::layer_param_t  gpu_h_input;
};


#endif
