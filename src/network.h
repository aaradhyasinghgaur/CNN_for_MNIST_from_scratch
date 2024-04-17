#ifndef NETWORK_H
#define NETWORK_H

#include "constants.h"                                     
#include "datahandler.h"                                     
#include "data.h"                                           
#include "convolution.h"                                                                          
                                          
#include "dense.h"                                                                                    
#include "activation.h"                                      
#include "activation_functions.h"                          

class Network {
public:
    Network() ;
    ~Network() = default;
    void read_training_dataset();
    void read_test_dataset();
    void training(double , double );
    void validation();
    void test();

    float accuracy_metric(Matrix_2d actual , Matrix_2d predicted);
    int argmax (const Vector_double& vec);


private:
    Convolution *c1;
    Sigmoid *s1;
    Dense *d1;
    Sigmoid *s2;
    Dense *d2;
    Softmax *s3;
    
    std::vector<Layer*> layers ;
    data_handler *dh;

};







#endif