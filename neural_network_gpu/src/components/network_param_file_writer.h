#ifndef _NETWORK_PARAM_FILE_WRITER_H
#define _NETWORK_PARAM_FILE_WRITER_H
/*********************************************************/
#include <memory>
#include "../layers/layer.h"
/*********************************************************/
class NetworkFileWriter
{
public:
  void UpdateModelFromFile(const std::string& path_to_file, std::vector<std::shared_ptr<Layer>>& layer_group);
  void SaveModelToFile(const std::string& path_to_save, std::vector<std::shared_ptr<Layer>>& layer_group);

private:
  int  ConvertFromBigEndian(unsigned char *buf);
  void ConvertToBigEndian(unsigned char *buf, const int val);

public:
  enum
  {
    FILE_MAJOR_VERSION = 1,
    FILE_MINOR_VERSION = 0,
  };

private:
  typedef enum
  {
    PARAM_TYPE_FLOAT  = 0,
    PARAM_TYPE_DOUBLE = 1,
  }Param_Type_e;

};



/*********************************************************/
#endif /* _NETWORK_PARAM_FILE_WRITER_H */
