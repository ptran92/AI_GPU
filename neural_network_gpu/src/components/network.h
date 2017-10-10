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
  void Loss(const float * neural_out, const float * expect_out, float * loss, int n);
  float * Forward_Propagate(float * input);

private:
  std::vector<std::shared_ptr<Layer>> layers;
  int       in_size;
  int       out_size;
  float     eta;
  int       batch_size;
  int       epoch_time;
  float *   gpu_input;
  float *   loss_dvt;
};


#endif
