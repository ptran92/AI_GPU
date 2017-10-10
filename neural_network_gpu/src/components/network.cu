/*************************************************************
*   File: network.cu
*
*
*************************************************************/
#include <cmath>
#include "network.h"
#include "device.h"
#include "cuda_runtime.h"
#include "helper.h"
/*************************************************************
 *    PUBLIC FUNCTIONS
 *************************************************************/
Network::Network(std::vector<std::shared_ptr<Layer>>& group_l, int input_size, int output_size, float lr, int b_size, int epoch):
  layers(group_l),
  in_size(input_size),
  out_size(output_size),
  eta(lr),
  batch_size(b_size),
  epoch_time(epoch)
{
  Helper::cuda_array_zero_allocate(&loss_dvt  , sizeof(float) * output_size);
  Helper::cuda_array_zero_allocate(&gpu_input , sizeof(float) * in_size);
}

Network::~Network()
{
  cudaFree(loss_dvt);
  cudaFree(gpu_input);
}

void Network::Predict(const float * input, float * output)
{
  // Copy to GPU memory
  cudaMemcpy(gpu_input, input, sizeof(float) * in_size, cudaMemcpyHostToDevice);

  // Feed the model
  float * gpu_output;
  gpu_output = Forward_Propagate(gpu_input);

  // Copy back to cpu buffer
  cudaMemcpy(output, gpu_output, sizeof(float) * out_size, cudaMemcpyDeviceToHost);

}

void Network::Train(const float * input, const float * e_output,  int total_train_samples,
            const float * test_input, const float * test_e_output, int total_test_samples)
{
  std::cout << "Start training....." << std::endl;
  std::cout << "+ Input size    : " << in_size << std::endl;
  std::cout << "+ Output size   : " << out_size << std::endl;
  std::cout << "+ Learning rate : " << eta << std::endl;
  std::cout << "+ Batch size    : " << batch_size << std::endl;
  std::cout << "+ Epoch time    : " << epoch_time << std::endl;
  std::cout << "+ Total samples : " << total_train_samples << std::endl;

  // First, allocate memory in gpu to store a batch
  float * gpu_b_input;
  float * gpu_e_output;
  float * gpu_neural_output;
  Helper::cuda_array_allocate(&gpu_b_input      , sizeof(float) * in_size  * batch_size);
  Helper::cuda_array_allocate(&gpu_e_output     , sizeof(float) * out_size * batch_size);
  Helper::cuda_array_allocate(&gpu_neural_output, sizeof(float) * out_size * batch_size);

  // Allocate space to store all neural outputs of a single batch on CPU side
  std::unique_ptr<float> cpu_neural_output(new float[out_size * batch_size]);

  // For each epoch time, do back propagation on training set
  int total_batches = total_train_samples / batch_size;

  for(int ep = 0; ep < epoch_time; ep++)
  {
    for(int b_count = 0; b_count < total_batches; b_count++)
    {
      // In training set, select random batch to do back propagation
      int     b_idx                    = rand() % (total_batches);
      float * b_start_addr             = (float *)input + b_idx * batch_size * in_size;
      float * b_ex_output_start_addr   = (float *)e_output + b_idx * batch_size * out_size;

      // Copy selected batch to gpu memory
      cudaMemcpy(gpu_b_input  , b_start_addr          , sizeof(float) * in_size  * batch_size, cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_e_output , b_ex_output_start_addr, sizeof(float) * out_size * batch_size, cudaMemcpyHostToDevice);

      // For each training sample in a batch, do back propagation
      for(int s_idx = 0; s_idx < batch_size; s_idx++)
      {
        float * sample        = gpu_b_input   + s_idx * in_size;
        float * ex_output     = gpu_e_output  + s_idx * out_size;
        float * neural_output;

        // Do forward propagation
        neural_output = Forward_Propagate(sample);

        // Copy to GPU memory pool for neural output
        cudaMemcpy(gpu_neural_output + s_idx * out_size, neural_output, sizeof(float) * out_size, cudaMemcpyDeviceToDevice);

        // Calculate loss derivative
        Helper::Cross_Entropy_Loss_Derivative(neural_output, ex_output, loss_dvt, out_size);

        // Back propagate loss derivative through each layer
        float * err_signal = loss_dvt;
        for(auto iter = layers.rbegin(); iter != layers.rend(); iter++)
        {
          err_signal = iter->get()->backward_propagation(err_signal);
        }

      }

      // After each batch, update each layer with new weights adn biases
      for(auto l : layers)
      {
        l->update(eta, batch_size);
      }

      // On specific batch index, calculate the loss/training set and loss/test set
      if(b_count % 500 == 0)
      {
        /*************************
         *
         * Calculate loss on training set
         *
         *************************/
        float batch_loss = 0.0;
        cudaMemcpy(cpu_neural_output.get(), gpu_neural_output, sizeof(float) * out_size * batch_size, cudaMemcpyDeviceToHost);
        Loss(cpu_neural_output.get(), b_ex_output_start_addr, &batch_loss, out_size * batch_size);

        std::cout << "\tLoss:  + Train: " << batch_loss << std::endl;

        // /*************************
        //  *
        //  * Calculate loss on validation set
        //  *
        //  *************************/
        // // select random batch in validation set
        // For test set
        // int total_test_batches = total_test_samples / batch_size;
        // int test_b_idx = rand() % total_test_batches;
        // float * test_b_start_addr             = (float *)test_input + test_b_idx * batch_size * in_size;
        // float * test_b_ex_output_start_addr   = (float *)test_e_output + test_b_idx * batch_size * out_size;
        //
        // // Copy selected batch to gpu memory
        // cudaMemcpy(gpu_b_input  , test_b_start_addr          , sizeof(float) * in_size  * batch_size, cudaMemcpyHostToDevice);
        // cudaMemcpy(gpu_e_output , test_b_ex_output_start_addr, sizeof(float) * out_size * batch_size, cudaMemcpyHostToDevice);
        //
        // // For each training sample in a batch, do back propagation
        // for(int s_idx = 0; s_idx < batch_size; s_idx++)
        // {
        //   float * sample        = gpu_b_input   + s_idx * in_size;
        //   float * neural_output;
        //
        //   // Do forward propagation
        //   neural_output = Forward_Propagate(sample);
        //
        //   // Copy to GPU memory pool for neural output
        //   cudaMemcpy(gpu_neural_output + s_idx * out_size, neural_output, sizeof(float) * out_size, cudaMemcpyDeviceToDevice);
        // }
        //
        // float test_loss = 0.0;
        // cudaMemcpy(cpu_neural_output.get(), gpu_neural_output, sizeof(float) * out_size * batch_size, cudaMemcpyDeviceToHost);
        // Loss(cpu_neural_output.get(), test_b_ex_output_start_addr, &test_loss, out_size * batch_size);
        //
        // std::cout << " - Validate: " << test_loss << std::endl;


      }
    }
  }

  std::cout << "Finish training" << std::endl;

  // Finish training, free GPU memory
  cudaFree(gpu_b_input);
  cudaFree(gpu_e_output);
}


/*************************************************************
 *  PRIVATE FUNCTIONS
 *************************************************************/
float * Network::Forward_Propagate(float * input)
{
  float *layer_feed = input;

  for(auto l : layers)
  {
    layer_feed = l->forward_propagation(layer_feed);
  }

  return layer_feed;
}

void Network::Loss(const float * neural_out, const float * expect_out, float * loss, int n)
{
  float sum = 0.0;
  for(int i = 0; i < n; i++)
  {
    sum += -( expect_out[i] * log(neural_out[i]) + (1 - expect_out[i]) * log(1 - neural_out[i]) );
  }

  *loss = sum / n;
}
