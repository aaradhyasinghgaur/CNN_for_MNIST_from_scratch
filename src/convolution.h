#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "layers.h"

class Convolution : public Layer
{
 public:

 Convolution(Vector_int, Vector_int);
 ~Convolution () = default;
 virtual Vector_double forward (const Vector_double& input_) override;
 virtual Vector_double backward (const Vector_double& output_gradient , double learning_rate ) override;

 private:

 Matrix_3d  self_input ;
 Matrix_4d  self_filters;  //4d
 Matrix_3d  self_biases ;  //3d

 Vector_int input_dim;   
 Vector_int kernel_dim ; 
 Vector_int output_dim  ; 

 int stride = 1 ;
 int padding = 1;
 int length = 0 ;
 int learning_rate = 0.001 ;
 int number_of_kernels ; 
 int filter_size ;
};


#endif