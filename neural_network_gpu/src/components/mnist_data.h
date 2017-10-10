#ifndef _MNIST_DATA_H
#define _MNIST_DATA_H

#include <iostream>
#include <fstream>
#include <memory>

class Mnist_Parser
{
public:
  Mnist_Parser(const std::string& image_file_name, const std::string& label_file_name)
  {
    Load_Data(image_file_name);
    Load_Label(label_file_name);
  }
private:
  void Load_Data(const std::string& file_name)
  {
    std::fstream ifs(file_name.c_str(), std::ios::in | std::ios::binary);

    if(!ifs.is_open())
    {
      std::cerr << "Cannot open file " << file_name << std::endl;
      exit(1);
    }

    unsigned char b[4];

    ifs.read((char *)b, sizeof(char) * 4);

    ifs.read((char *)b, sizeof(char) * 4);
    int n_image = big_endian(b);
    std::cout << "Total test images: " << n_image << std:: endl;

    ifs.read((char *)b, sizeof(char) * 4);
    int rows = big_endian(b);
    std::cout << "Image size: " << rows;

    ifs.read((char *)b, sizeof(char) * 4);
    int cols = big_endian(b);
    std::cout << " x " << cols << std::endl;

    std::shared_ptr<float> data_pool(new float[rows * cols * n_image]);
    unsigned char* buf = new unsigned char[rows * cols];
    float *pool_adr = data_pool.get();

    for(int curr_img = 0, n_pixel = rows * cols; curr_img < n_image; curr_img++)
    {
      ifs.read((char *)buf, sizeof(char) * n_pixel);

      // Normalize data to avoid neural saturation
      for(int i = 0, img_adr = curr_img * n_pixel; i < n_pixel; i++)
        pool_adr[img_adr + i] = (buf[i] - 128.0) / 128.0;
    }

    delete[] buf;

    ifs.close();

    /* Save away image data */
    image       = data_pool;
    total_image = n_image;
    img_rows    = rows;
    img_cols    = cols;
  }

  void Load_Label(const std::string& file_name)
  {
    std::fstream ifs(file_name.c_str(), std::ios::in | std::ios::binary);

    if(!ifs.is_open())
    {
      std::cerr << "Cannot open file " << file_name << std::endl;
      exit(1);
    }

    unsigned char b[4];

    ifs.read((char *)b, sizeof(char) * 4);

    ifs.read((char *)b, sizeof(char) * 4);
    int n_image = big_endian(b);


    std::shared_ptr<float> class_pool(new float[10 * n_image]);
    float *pool_adr = class_pool.get();

    // Clear all labels
    for(int i = 0; i < (10 * n_image); i++)
    {
      pool_adr[i] = 0;
    }

    for(int curr_img = 0; curr_img < n_image; curr_img++)
    {
      unsigned char true_idx;
      ifs.read((char *)&true_idx, sizeof(char));

      pool_adr[curr_img * 10 + true_idx] = 1.0;
    }

    ifs.close();

    /* Save away information on labels */
    label       = class_pool;
    total_label = n_image;
    total_class = 10;

  }

  int big_endian(unsigned char* c)
  {
    int ret = 0;

    ret |= c[0] << 24;
    ret |= c[1] << 16;
    ret |= c[2] << 8;
    ret |= c[3];

    return ret;
  }
public:
  /* Pointer to images & labels storage */
  std::shared_ptr<float> image;
  std::shared_ptr<float> label;
  /* Information about images */
  int total_image;
  int img_rows;
  int img_cols;
  /* Information about labels */
  int total_label;
  int total_class;
};


#endif
