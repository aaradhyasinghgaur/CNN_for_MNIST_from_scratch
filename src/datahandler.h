// splitting data. validation data - to check if model is trained according to our liking. (threshold)
//function - reading data , splitting data , getting splitted data.

#ifndef DATAHANDLER_H
#define DATAHANDLER_H


#include "constants.h"
#include "data.h"


class data_handler
{
public:
   data_handler () ;
   ~data_handler () ;

    //after the data file is seperated we'll read them seperately.
   void read_feature_vector (std::string path);
   void read_feature_label (std::string path);
   void split_data (/*std::vector <uint8_t> * */);
   void count_classes();

   uint32_t convert_to_little_endian(const unsigned char *);
 
   //getter functions.
   std::vector <data*>* get_data_array();
   std::vector <data*>* get_training_data ();
   std::vector <data*>*  get_test_data ();
   std::vector <data*>*  get_validation_data ();


     int random_number(int min, int max) 
    {
    static std::mt19937 rng(std::random_device{}()); 
    return std::uniform_int_distribution<int>{min, max}(rng); 
    }


private:
   std::vector <data*> *data_array ; // all of the data (pre-split)
   std::vector <data*> *training_data;
   std::vector <data*> *validation_data ;
   std::vector <data*> *test_data;

   int num_classes ; // number of classes.
   int feature_vactor_size ;
//    std::map<uint8_t , int> class_map;

   const double TRAINING_SET_PERCENT = 0.90;
   const double TEST_SET_PERCENT = 0;
   const double VALID_SET_PERCENT = 0.10;
    
};
#endif