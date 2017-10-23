/*************************************************************
*   File: main.cu
*
*
*************************************************************/
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include "src/components/network.h"
#include "src/components/device.h"
#include "src/layers/layer.h"
#include "src/layers/fc_layer.h"
#include "src/layers/softmax_layer.h"
#include "src/components/mnist_data.h"
#include "src/components/network_param_file_writer.h"
/*************************************************************
 *    MACROS & DEFINITIONS
 *************************************************************/
#define TRAIN_MODE      0
#define INFERENCE_MODE  1

/*************************************************************
 *    CONSTANTS
 *************************************************************/
/* Data path */
#define TRAINING_DATA_PATH        "data/train-images.idx3-ubyte"
#define TRAINING_LABEL_PATH       "data/train-labels.idx1-ubyte"
#define VALIDATE_DATA_PATH        "data/t10k-images.idx3-ubyte"
#define VALIDATE_LABEL_PATH       "data/t10k-labels.idx1-ubyte"

/* Model path */
#define MODEL_PATH_FILE           "model/trained_model.txt"

/* network parameter */
#define BATCH_SIZE                        20
#define LEARNING_RATE                     0.1
#define EPOCH_TIME                        10
/*************************************************************
 *    MODULE FUNCTIONS
 *************************************************************/
void Train_Model(Network& net, Mnist_Parser& training_set, Mnist_Parser& test_set)
{
  // Start training
  net.Train(training_set.image.get(), training_set.label.get(), training_set.total_image,
              test_set.image.get(), test_set.label.get(), test_set.total_image);

}

void Validate_Model(Network& net, Mnist_Parser& test_set)
{
  std::shared_ptr<float> single_neural_output(new float[test_set.total_class]);
  int total_test_images = test_set.total_image;
  int total_pxs         = test_set.img_rows * test_set.img_cols;
  int total_class       = test_set.total_class;
  float *image         = test_set.image.get();
  float *res           = test_set.label.get();
  float accuracy       = 0.0;

  std::cout << "/**************** TEST RESULT ****************/" << std::endl;

  for(int i = 0; i < total_test_images; i++)
  {
    float *input          = image + (i * total_pxs);
    float *e_output       = res + (i * total_class);
    float *neural_output  = single_neural_output.get();

    // Feed the network with test data
    net.Predict(input, neural_output);

    // Calculate error percentage
    int     predict_num      = 0;
    int     true_num         = 0;
    float   predict_max_prob = neural_output[0];
    float   true_max_prob    = e_output[0];


    for(int c = 0; c < total_class; c++)
    {
      if(predict_max_prob < neural_output[c])
      {
        predict_max_prob = neural_output[c];
        predict_num      = c;
      }

      if(true_max_prob < e_output[c])
      {
        true_max_prob = e_output[c];
        true_num      = c;
      }
    }

    if(predict_num == true_num)
    {
      accuracy += 1.0;
    }
  }

  std::cout << "Average accuracy: " << (accuracy * 100.0 / total_test_images) << std::endl;

}

/*************************************************************
 *    MAIN
 *************************************************************/
int main(int argc, char const *argv[])
{
  /*****************************************************
   *  Read MNIST data
   *****************************************************/
  Mnist_Parser training_set(TRAINING_DATA_PATH, TRAINING_LABEL_PATH);
  Mnist_Parser test_set(VALIDATE_DATA_PATH, VALIDATE_LABEL_PATH);

  int input_size  = training_set.img_rows * training_set.img_cols;
  int output_size = training_set.total_class;

  /*****************************************************
   *  Create the model
   *****************************************************/
  std::vector<std::shared_ptr<Layer>> group_layers(3);
  group_layers[0] = std::make_shared<FC_Layer>(input_size, 64);
  group_layers[1] = std::make_shared<FC_Layer>(64, 64);
  group_layers[2] = std::make_shared<Softmax_Layer>(64, output_size);

  // create handle for cublas
  Device::Device_Create();

  /*****************************************************
   *  Create network
   *****************************************************/
  Network net(group_layers, input_size, output_size, LEARNING_RATE, BATCH_SIZE, EPOCH_TIME);


  #if TRAIN_MODE
  /*****************************************************
   *  Train
   *****************************************************/
  std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
  Train_Model(net, training_set, test_set);
  std::chrono::system_clock::time_point end   = std::chrono::system_clock::now();

  std::chrono::milliseconds elapsed_millisecs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  /* Print the elapsed time */
  std::cout << "Time elapsed: " << (double)(elapsed_millisecs.count())/1000/60 << " minutes" << std::endl;

  /* Save to file */
  NetworkFileWriter writer;
  std::string save_path = MODEL_PATH_FILE;
  std::cout << "Save file to " << save_path << std::endl;
  writer.SaveModelToFile(save_path, group_layers);
  #endif /* TRAIN_MODE */

  #if INFERENCE_MODE
  /*****************************************************
   *  Load model from file
   *****************************************************/
  NetworkFileWriter writer;
  std::string load_path = MODEL_PATH_FILE;
  std::cout << "Read file from " << load_path << std::endl;
  writer.UpdateModelFromFile(load_path, group_layers);
  #endif /* INFERENCE_MODE */
  /*****************************************************
   *  Validate
   *****************************************************/
  Validate_Model(net, test_set);

  /*****************************************************
   *  End of application
   *****************************************************/
  // remove cublas handle
  Device::Device_Destroy();
  return 0;
}
