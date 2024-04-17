// we need to split data randomly as to get random pointers of data stored in different sites.
#include "datahandler.h"          

    data_handler::data_handler()                                                          
    {
        data_array = new std::vector<data*>;                    
        training_data = new std::vector<data*>;                         
        test_data = new std::vector<data*>;                              
        validation_data = new std::vector<data*>;                       
    }

    data_handler::~data_handler()                               
    {
        //free dynamically allocated data.    
        delete data_array;     
        delete training_data;
        delete test_data;
        delete validation_data;    
    }

   void data_handler::read_feature_vector (std::string path)                
   {
        uint32_t header[4];      /*data will be stored in low endian.*/           /*0000     32 bit integer  0x00000803(2051) magic number
                                                                               0004     32 bit integer  60000            number of images
                                                                               0008     32 bit integer  28               number of rows
                                                                               0012     32 bit integer  28               number of columns */
    
        unsigned char bytes[4];         // data stored in high endian.          
        FILE *file = fopen(path.c_str() , "rb");  // should be rb? 
        // std::ifstream file(path, std::ios::binary);                   

        if (file)        
        {
        //for header . 
        for (int i = 0 ; i < 4 ; i++)                                      
        {
            //read 1 element of size(bytes) each from "file" to "bytes".
            if (fread(bytes , sizeof(bytes) , 1 , file)) 
            {
                header[i] = convert_to_little_endian(bytes);                 
            }
        }

        std::cout << "Done getting file header (integer number). " << std::endl;

        std::cout << "print file header :" << std::endl;
        for (int i = 0 ; i < 4 ; ++i)
        {
            std::cout << header[i] << std::endl;
        }

        int image_size = header[2] * header[3];
        for (size_t i = 0 ; i < header[1] ; i++)
        {
            //setting up new data container -> feature vector , label.
            data *d = new data();
            uint8_t element[1] ;     //should be element[1]?
            for (int j = 0 ; j < image_size ; ++j)
            {
                //read 1 element of size(elment) each from file to element.
                if (fread(element, sizeof(element), 1, file))
                {
                    unsigned int decimalValue = 0;
                    for (int bit = 0; bit < 8; ++bit) {
                        if (element[0] & (1 << bit)) {
                            decimalValue += (1 << bit);
                        }
                    }
                d->append_to_feature_vector(decimalValue);

                }else 
                { 
                    std::cerr << "ERROR: cannot read from file." << std::endl;
                    exit(1);
                }
            }
            data_array->push_back(d);
        }

        std::cout << "Successfully read and stored feature vectors." << std::endl;
        }
        else
        {
            std::cout << "ERROR : could not find file." << std::endl;   
            exit(1); 
        }
   }

   void data_handler::read_feature_label (std::string path)                                   
   {
        uint32_t header[2];  // two 32 bit integers. (low endian).
        unsigned char bytes[4]; //high endian

        FILE *f = fopen (path.c_str() , "rb");  //should be rb?
        if (!f)
        {
            std::cerr << "ERROR: file cannot open." << std::endl;
            exit(1);
        }

        //created header. {magic_num ,number_of_items}
        for (int i = 0 ; i < 2 ; ++i)
        {
            //read 1 element of size(bytes) from f to bytes.
            if(fread(bytes , sizeof(bytes) , 1 , f))
            {
              header[i] = convert_to_little_endian(bytes);
            }
        }
        std::cout << "Done getting label file header." << std::endl;

        std::cout << "print label header:" << std::endl;
        for (int i = 0 ; i < 2 ;++i)
        {
            std::cout << header[i] << std::endl;
        }

        //appending data to label.
        for (size_t i = 0 ; i < header[1] ; ++i)
        {
            //setting up new data container -> feature vector , label.
            uint8_t element[1];
            //read 1 element of size(elment) each from file to element.
            if(fread(element , sizeof(element) , 1 , f))
            {
                    unsigned int decimalValue = 0;
                    for (int bit = 0; bit < 8; ++bit) {
                        if (element[0] & (1 << bit)) {
                            decimalValue += (1 << bit);
                        }
                    }
                 data_array->at(i)->set_label(decimalValue);  
            }else 
            { 
                std::cerr << "ERROR: cannot read from file." << std::endl;
                exit(1);
            }
        }

        std::cout << "Successfully read and stored labels." << std::endl;
   }


   void data_handler::split_data ()                       
   {
        std::unordered_set<int> used_index;
        int size = data_array->size(); //sizeof(data_array) will give size of std::vector<data>.
        int training_size = size * TRAINING_SET_PERCENT;
        int valid_size = size * VALID_SET_PERCENT;
        int test_size = size * TEST_SET_PERCENT;

        std::cout << "Size of data_array: " << size <<  std::endl;
        std::cout << "Training data size : " << training_size <<  std::endl;
        std::cout << "test data size : " << test_size <<  std::endl;
        std::cout << "validation data size : " << valid_size <<  std::endl;

        if ((size != 0) || (size>training_size  &&  size>test_size  && size>valid_size))
        {
        //Training Data
        int count = 0 ;
        while(count < training_size)
        {
            int rand_index = random_number(0 , size-1);
            if(used_index.find(rand_index) == used_index.end())  //jab random index element nahi mila.
            {
                training_data->push_back(data_array->at(rand_index));
                used_index.insert(rand_index);
                count++;
            }
        }

        std::cout << "done splitting training data. " << std::endl;
 
         //Validation Data
        count = 0 ;
        while(count < valid_size)
        {
            int rand_index = random_number(0 , size-1);
            if(used_index.find(rand_index) == used_index.end())  //jab random index element nahi mila.
            {
                validation_data->push_back(data_array->at(rand_index));
                used_index.insert(rand_index);
                count++;
            }
        }

        std::cout << "done splitting validation data. " << std::endl;

        //Test Data
        count = 0 ;
        while(count < test_size)
        {
            int rand_index = random_number(0 , size-1);
            if(used_index.find(rand_index) == used_index.end())  //jab random index element nahi mila.
            {
                test_data->push_back(data_array->at(rand_index));
                used_index.insert(rand_index);
                count++;
            }
        }

         std::cout << "done splitting test data. " << std::endl;
        }

        std::cout << "Training data size : " << training_data->size() << std::endl;
        std::cout << "Validation data size : " <<validation_data->size() << std::endl;
        std::cout << "Test data size : " << test_data->size() << std::endl;
   }

   //can use mapping as well.
   void data_handler::count_classes()                                    
   {
    std::unordered_set <unsigned int> already_avail_labels;
    std::unordered_set <int> used_index;
    int size = data_array->size();

    num_classes = 0;

    for (int count = 0 ; count < size ; ++count)
    {
        int rand_index = rand() % size;
        if (used_index.find(rand_index) == used_index.end())
        {
            used_index.insert(rand_index);
        if (already_avail_labels.find(data_array->at(rand_index)->get_label()) == already_avail_labels.end())
        {
          already_avail_labels.insert(data_array->at(rand_index)->get_label());
          num_classes++;
        }
      }
    }

      std::cout << "Succesfully extracted number of classes : " << num_classes << std::endl;
   }

   uint32_t data_handler::convert_to_little_endian(const unsigned char *bytes)                  
   {
         return (uint32_t) ((bytes[0] << 24) |
                             (bytes[1] << 16) |
                             (bytes[2] << 8) |
                             (bytes[3]));
   }

   std::vector <data*>* data_handler::get_data_array()
   {
     return data_array;
   }

   std::vector <data*>* data_handler ::get_training_data ()
   {
     return training_data;
   }
   std::vector <data*>* data_handler ::get_test_data ()
   {
     return test_data;
   }

   std::vector <data*>* data_handler ::get_validation_data ()
   {
      return validation_data;
   }



