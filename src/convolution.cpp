#include "convolution.h"
#include "matrix.h"


Convolution::Convolution(Vector_int _input_dim , Vector_int _kernel_dim)
{
   //initializing input_dim.
  for (int i = 0 ; i < _input_dim.size() ; ++i)
  {
     input_dim.push_back(_input_dim[i]) ;                    // {1 , 28 , 28 }
  }
  

  //initializing kernel_dim.
  for (int i = 0 ; i < _kernel_dim.size() ; ++i)
  {
     kernel_dim.push_back(_kernel_dim[i]);                    // {5 , 1 , 3 , 3}
  }
           

  //initializing output_dim.
  output_dim.push_back(kernel_dim[0])  ;                              // 5
  output_dim.push_back(input_dim[1] - kernel_dim[2] + 1 );               //  28 - 3 + 1 = 26
  output_dim.push_back (input_dim[2] - kernel_dim[3] + 1 );               //  28 - 3 + 1 = 26

  
  self_filters = initializeWeights(kernel_dim);              
  self_biases = set_random_matrix_3d(output_dim);


}


 Vector_double Convolution::forward (const Vector_double& input_ ) 
 {
   // converting input_ 1-d vector to 3_d vector for easy calculation.
   Matrix_3d result(output_dim[0], Matrix_2d( output_dim[1], Vector_double(output_dim[2] , 0.00) ) );
  
   self_input.clear();

   if (input_dim.size() == 3)
   {
      self_input = reshape_matrix_to_3d(input_dim , input_);
   }
   
   for (int i = 0 ; i < output_dim[0] ; ++i)  {     //5
      for (int j = 0 ; j < output_dim[1] ; ++j)  {       //26
         for (int k = 0 ; k < output_dim[2] ; ++k)  {       //26
            double sum = 0.0000000 ;
            for (int l = 0 ; l < input_dim[0] ; ++l)  {         //1
               for (int m = 0 ; m < kernel_dim[2] ; ++m)  {     //3
                  for (int n = 0 ; n < kernel_dim[3] ; ++n)  {      //3
                  sum += self_input[l][j * stride + m][k * stride + n] * self_filters[i][l][m][n];   // 5 , 1, 3 , 3
                 }  
               } 
            }
          result[i][j][k] = sum + self_biases[i][j][k];  
         }  
      }
   }

   
   Vector_double output = reshape_matrix_to_1d(output_dim , result);

   return output;
 }


 Vector_double Convolution::backward(const Vector_double& output_gradient
                            , double learning_rate ) 
{
      // 3d input_gardient = 0 (initialization)
   Matrix_3d input_gradient( input_dim[0], Matrix_2d(input_dim[1], Vector_double(input_dim[2] , 0) ) );
   Matrix_3d grad_input( output_dim[0], Matrix_2d(input_dim[1], Vector_double(input_dim[2] , 0) ) );

      // // Update weights using L2 regularization
      // for (size_t i = 0; i < self_filters.size(); ++i) {
      //       for (size_t j = 0; j < self_filters[i].size(); ++j) {
      //          for (size_t k = 0; k < self_filters[i][j].size(); ++k) {
      //             for (size_t m = 0; m < self_filters[i][j][k].size(); ++m) {
      //             self_filters[i][j][k][m] -= lambda * self_filters[i][j][k][m];
      //          }
      //       }
      //    }
      // }

      // 4d kernel_gardient = 0 (initialization)
   Matrix_4d kernel_gradient (kernel_dim[0], Matrix_3d( kernel_dim[1], Matrix_2d(kernel_dim[2], Vector_double(kernel_dim[3] , 0) ) ));    
   Matrix_3d reshape_output_grad ;
   Matrix_4d rotated_kernel;
   Matrix_3d padded_output_grad;
   
   reshape_output_grad.clear();
   reshape_output_grad = reshape_matrix_to_3d(output_dim , output_gradient);

   rotated_kernel.clear();
   rotated_kernel = rotate_matrix_180(self_filters);

   padded_output_grad.clear();
   padded_output_grad = padding_matrix(reshape_output_grad , padding = 2); //for input_grad.

   // calculating kernel gradient.  //checked (correct)
   for ( int i = 0 ; i < kernel_dim[0] ; ++i) {    //5
      for ( int j = 0 ; j < kernel_dim[1] ; ++j) {        //1
         for ( int k = 0 ; k < kernel_dim[2] ; ++k) {      //3  
            for ( int m = 0 ; m < kernel_dim[3] ; ++m) {    //3
               double sum = 0.0000000 ;
               for (int p = 0 ; p < input_dim[0] ; ++p) {       //1
                  for (int n = 0 ; n < output_dim[1] ; ++n){         //26
                     for (int r = 0; r < output_dim[2] ; ++r)  {     //26
                        sum += self_input[p][k * stride + n][m * stride + r] * reshape_output_grad[i][n][r]  ; 
                     }
                  } 
               }
              kernel_gradient[i][j][k][m] = sum ; 
            }  
         }
      }
   }

   //calculating input gradient.
    for (int i = 0 ; i < output_dim[0] ; ++i)  {                   
      for (int j = 0 ; j < input_dim[1] ; ++j)  {                   
         for (int k = 0 ; k < input_dim[2] ; ++k) {                  
            double sum = 0.0000000 ;
            for (int l = 0 ; l < input_dim[0] ; ++l){     
               for (int m = 0 ; m < kernel_dim[2] ; ++m){                 
                  for (int n = 0 ; n < kernel_dim[3] ; ++n){  
                     sum += padded_output_grad[i][j * stride + m][k * stride + n] * rotated_kernel[i][l][m][n];
                  } 
               }  
            }
            grad_input[i][j][k] = sum;
         }
      }
   }

   // changing (5 * 28 * 28) to (1 * 28 * 28)
   for (int m = 0; m < input_dim[0] ; ++m){     
   for (int i = 0 ; i < input_dim[1] ; ++i)  {            
      for (int j = 0 ; j < input_dim[2] ; ++j)  {               
         for (int k = 0 ; k < output_dim[0]; ++k) {    
         input_gradient[m][i][j] = grad_input[k][i][j] ;
         }
      }
   }
}

   // updating filters.
   for (int i = 0 ; i < self_filters.size() ; ++i){
      for (int j = 0 ; j < self_filters[0].size() ; ++j){
         for (int k = 0 ; k < self_filters[0][0].size() ; ++k){
            for (int l = 0 ; l < self_filters[0][0][0].size() ; ++l){
               self_filters[i][j][k][l] = self_filters[i][j][k][l] - (learning_rate * kernel_gradient[i][j][k][l]) ;
            }
         }
      }
   }

   // updating biases.
   for (int j = 0 ; j < self_biases.size() ; ++j){
      for (int k = 0 ; k < self_biases[0].size() ; ++k){
         for (int l = 0 ; l < self_biases[0][0].size() ; ++l){
            self_biases[j][k][l] -= learning_rate * reshape_output_grad[j][k][l] ;
         }
      }
   }

 
}