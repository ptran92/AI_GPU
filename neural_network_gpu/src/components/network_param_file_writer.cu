/*************************************************************
*   File: network_param_file_writer.cpp
*
*
*************************************************************/
#include <iostream>
#include <fstream>
#include "network_param_file_writer.h"
#include "cuda_runtime.h"

#if USING_HALF_FLOAT
#include "cuda_fp16.h"
#include "helper.h"
#endif /* USING_HALF_FLOAT */
/*************************************************************
 *    MACROS & CONSTANTS
 *************************************************************/
#define MEMPOOL_BYTE_SIZE      600 // NOTE: must be divided by 8, which is sizeof(double)

/*************************************************************
 *    PUBLIC FUNCTIONS
 *************************************************************/
 void NetworkFileWriter::UpdateModelFromFile(const std::string& path_to_file, std::vector<std::shared_ptr<Layer>>& layer_group)
 {
   std::fstream ifs(path_to_file.c_str(), std::ios::in | std::ios::binary);

   if( !ifs.is_open() )
   {
     std::cerr << "Cannot open model file: " << path_to_file << std::endl;
     exit(1);
   }

   unsigned char read_bytes[4];
   unsigned char major_version;
   unsigned char minor_version;
   Param_Type_e  param_type;
   int           param_size;
   unsigned char total_layers;

   /* Read file version */
   ifs.read((char*)read_bytes  , 1);
   major_version = read_bytes[0];

   ifs.read((char*)read_bytes  , 1);
   minor_version = read_bytes[0];

   std::cout << "File version " << major_version << "." << minor_version << std::endl;

   /* Read the type of model parameter and total layers the file stored */
   ifs.read((char*)read_bytes  , 1);
   param_type    = (Param_Type_e)read_bytes[0];

   ifs.read((char*)read_bytes  , 1);
   total_layers  = read_bytes[0];

   if( total_layers != layer_group.size() )
   {
     std::cerr << "Model not matched" << std::endl;
     exit(1);
   }

   if( param_type == PARAM_TYPE_FLOAT )
   {
     param_size = sizeof(float);
   }
   else if( param_type == PARAM_TYPE_DOUBLE )
   {
     param_size = sizeof(double);
   }
   else
   {
     std::cerr << "Cannot get size of param type: " << param_type << std::endl;
     exit(1);
   }

   /* Parse parameters for each layer */
   for(auto l : layer_group)
   {
     int     layer_weight_byte_size;
     int     layer_bias_byte_size;
     int     layer_input_size   = l->GetInputSize();
     int     layer_output_size  = l->GetOutputSize();
     int     no_weight_elements = layer_output_size * layer_input_size;
     int     no_bias_elements   = layer_output_size;

     ifs.read((char*)read_bytes, 4);// this variable is not used rightnow

     ifs.read((char*)read_bytes, 4);
     layer_weight_byte_size = ConvertFromBigEndian(read_bytes);

     ifs.read((char*)read_bytes, 4);
     layer_bias_byte_size = ConvertFromBigEndian(read_bytes);

     /* Sanity check */
     if(
        ( (layer_weight_byte_size/param_size) != (no_weight_elements) ) ||
        ( (layer_bias_byte_size/param_size)   != (no_bias_elements) )
        )
     {
       std::cerr << "Unmatched weight/bias size" << std::endl;
       exit(1);
     }

     if( param_type == PARAM_TYPE_FLOAT )
     {
       int mempool_total_elements = MEMPOOL_BYTE_SIZE / param_size;
       std::unique_ptr<float []> mem_pool( new float[ mempool_total_elements ] );
       int offset;
       int remaining;
       int idx;

       #if USING_HALF_FLOAT
         /* Convert to GPU memory */
         float * f_gpu_w;
         float * f_gpu_b;
         layer_param_t l_w = l->GetWeight();
         layer_param_t l_b = l->GetBias();

         /* Allocate temporary float array, in order to convert to half float later */
         Helper::cuda_array_allocate(&f_gpu_w, Layer::FLOAT_TYPE, no_weight_elements);
         helper::cuda_array_allocate(&f_gpu_b, Layer::FLOAT_TYPE, no_bias_elements);

         /* Read weight from file */
         idx        = 0;
         offset     = 0;
         remaining  = layer_weight_byte_size / param_size;
         while(remaining > 0)
         {
           /* read an element from file and save to buffer */
           ifs.read((char*)read_bytes, 4);
           mem_pool[idx] = (float)ConvertFromBigEndian(read_bytes);

           /* Just consumed an element */
           remaining--;
           idx++;

           /* if buffer is full or no more data to read, flush buffer to GPU memory */
           if( idx == mempool_total_elements ||
               remaining == 0 )
           {
             /* Copy to GPU memory */
             cudaMemcpy(f_gpu_w + offset, mem_pool.get(), param_size * idx, cudaMemcpyHostToDevice);
             offset += idx;

             /* Reset counter */
             idx = 0;
           }
         }

         /* Read bias from file */
         offset     = 0;
         idx        = 0;
         remaining  = layer_bias_byte_size / param_size;
         while(remaining > 0)
         {
           /* read an element from file and save to buffer */
           ifs.read((char*)read_bytes, 4);
           mem_pool[idx] = (float)ConvertFromBigEndian(read_bytes);

           /* Just consumed an element */
           remaining--;
           idx++;

           /* if buffer is full or no more data to read, flush buffer to GPU memory */
           if( idx == mempool_total_elements ||
               remaining == 0 )
           {
             /* Copy to GPU memory */
             cudaMemcpy(f_gpu_b + offset, mem_pool.get(), param_size * idx, cudaMemcpyHostToDevice);
             offset += idx;

             /* Reset counter */
             idx = 0;
           }
         }

         /* Convert to half float weight in GPU */
         Helper::cvtfloat2half(f_gpu_w, l_w, no_weight_elements);

         /* Convert to half float bias in GPU */
         Helper::cvtfloat2half(f_gpu_b, l_b, no_bias_elements);

         /* Free memory */
         cudaFree(f_gpu_w);
         cudaFree(f_gpu_b);
       #else /* not USING_HALF_FLOAT */
         /* Convert to GPU memory */
         Layer::layer_param_t l_w = l->GetWeight();
         Layer::layer_param_t l_b = l->GetBias();

         /* Read weight from file */
         idx        = 0;
         offset     = 0;
         remaining  = layer_weight_byte_size / param_size;
         while(remaining > 0)
         {
           /* read an element from file and save to buffer */
           ifs.read((char*)read_bytes, 4);
           mem_pool[idx] = (float)ConvertFromBigEndian(read_bytes);

           /* Just consumed an element */
           remaining--;
           idx++;

           /* if buffer is full or no more data to read, flush buffer to GPU memory */
           if( idx == mempool_total_elements ||
               remaining == 0 )
           {
             /* Copy to GPU memory */
             cudaMemcpy(l_w + offset, mem_pool.get(), param_size * idx, cudaMemcpyHostToDevice);
             offset += idx;

             /* Reset counter */
             idx = 0;
           }
         }


         /* Read bias from file */
         offset     = 0;
         idx        = 0;
         remaining  = layer_bias_byte_size / param_size;
         while(remaining > 0)
         {
           /* read an element from file and save to buffer */
           ifs.read((char*)read_bytes, 4);
           mem_pool[idx] = (float)ConvertFromBigEndian(read_bytes);

           /* Just consumed an element */
           remaining--;
           idx++;

           /* if buffer is full or no more data to read, flush buffer to GPU memory */
           if( idx == mempool_total_elements ||
               remaining == 0 )
           {
             /* Copy to GPU memory */
             cudaMemcpy(l_b + offset, mem_pool.get(), param_size * idx, cudaMemcpyHostToDevice);
             offset += idx;

             /* Reset counter */
             idx = 0;
           }
         }

       #endif

     }
     else if( param_type == PARAM_TYPE_DOUBLE)
     {
       // TODO:
     }
   }

   /* Finish reading */
   ifs.close();
 }
 void NetworkFileWriter::SaveModelToFile(const std::string& path_to_save, std::vector<std::shared_ptr<Layer>>& layer_group)
 {
#if USING_HALF_FLOAT
    // NOTE: saving model to file is not supported for half float

#else /* not USING_HALF_FLOAT */
   std::fstream ofs(path_to_save.c_str(), std::ios::out | std::ios::binary);

   if( !ofs.is_open() )
   {
     std::cerr << "Cannot open file: " << path_to_save << std::endl;
     exit(1);
   }

   unsigned char bytes_write[4];
   int param_size;
   /* Write header to file */
   bytes_write[0] = FILE_MAJOR_VERSION;
   bytes_write[1] = FILE_MINOR_VERSION;
   bytes_write[2] = (unsigned char)PARAM_TYPE_FLOAT;
   bytes_write[3] = layer_group.size();
   ofs.write((char *)bytes_write, 4);

   /* Handle param tytpe float / double */
   param_size = sizeof(float); // NOTE: Hardcoded for now

   /* For each layer, write its weight and bias */
   for(auto l : layer_group)
   {
     int total_weight_element = l->GetInputSize() * l->GetOutputSize();
     int total_bias_element   = l->GetOutputSize();

     /* Write byte size of weight and bias */
     ConvertToBigEndian(bytes_write, total_weight_element * param_size);
     ofs.write((char *)bytes_write, 4);
     ConvertToBigEndian(bytes_write, total_bias_element   * param_size);
     ofs.write((char *)bytes_write, 4);

     /* Process weight and bias */
     Layer::layer_param_t l_w = l->GetWeight();
     Layer::layer_param_t l_b = l->GetBias();
     int offset;
     int elements_to_copy;
     int remaining_elements;
     int mempool_total_elements = MEMPOOL_BYTE_SIZE / param_size;
     std::unique_ptr<float []> mem_pool( new float[ mempool_total_elements ] );

     // Read weights from GPU memory into buffer and write to file
     offset = 0;
     remaining_elements = total_weight_element;
     while(remaining_elements > 0)
     {
       /* Copy elements from GPU memory to buffer */
       elements_to_copy = (remaining_elements > mempool_total_elements)? mempool_total_elements : remaining_elements;
       cudaMemcpy(mem_pool.get(), l_w + offset, elements_to_copy * param_size, cudaMemcpyDeviceToHost);

       /* Convert from little endian to big endian */
       for(int element_idx = 0; element_idx < elements_to_copy; element_idx++)
       {
         float temp;
         ConvertToBigEndian((unsigned char * )&temp, mem_pool[element_idx]);
         mem_pool[element_idx] = temp;
       }

       /* Write copied elements to file */
       ofs.write((char *)mem_pool.get(), elements_to_copy * param_size);

       /* Just consumed some elements */
       remaining_elements -= elements_to_copy;
       offset += elements_to_copy;
     }

     // Read bias from GPU memory into buffer and write to file
     offset = 0;
     remaining_elements = total_bias_element;
     while(remaining_elements > 0)
     {
       /* Copy elements from GPU memory to buffer */
       elements_to_copy = (remaining_elements > mempool_total_elements)? mempool_total_elements : remaining_elements;
       cudaMemcpy(mem_pool.get(), l_b + offset, elements_to_copy * param_size, cudaMemcpyDeviceToHost);

       /* Convert from little endian to big endian */
       for(int element_idx = 0; element_idx < elements_to_copy; element_idx++)
       {
         float temp;
         ConvertToBigEndian((unsigned char * )&temp, mem_pool[element_idx]);
         mem_pool[element_idx] = temp;
       }

       /* Write copied elements to file */
       ofs.write((char *)mem_pool.get(), elements_to_copy * param_size);

       /* Just consumed some elements */
       remaining_elements -= elements_to_copy;
       offset += elements_to_copy;
     }
   }

   /* finish writing */
   ofs.close();
#endif /* USING_HALF_FLOAT */
 }
 /*************************************************************
  *   PRIVATE FUNCTIONS
  *************************************************************/
int NetworkFileWriter::ConvertFromBigEndian(unsigned char *buf)
{
  int temp = 0;

  temp |= ((int)buf[0]&0xFF) << 24;
  temp |= ((int)buf[1]&0xFF) << 16;
  temp |= ((int)buf[2]&0xFF) << 8;
  temp |= ((int)buf[3]&0xFF);

  return temp;
}
void NetworkFileWriter::ConvertToBigEndian(unsigned char *buf, const int val)
{
  buf[0] = (unsigned char)((val >> 24) & 0xFF);
  buf[1] = (unsigned char)((val >> 16) & 0xFF);
  buf[2] = (unsigned char)((val >> 8) & 0xFF);
  buf[3] = (unsigned char)(val & 0xFF);
}
 /*************************************************************
  *   END OF FILE
  *************************************************************/
