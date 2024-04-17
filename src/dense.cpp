#include "dense.h"
#include "matrix.h"


Dense::Dense(int input_size , int output_size) : input_size(input_size) , output_size(output_size)
{
    weights = initializeWeights_2d(output_size , input_size);
    bias =  Generate_Random_Vector(output_size);
}



Vector_double Dense::forward (const Vector_double& input) 
{
    self_input.clear();
    
    self_input = input ;
    
   Vector_double result (output_size , 0.00);
    for (size_t i = 0 ; i < output_size ; ++i) {       //10
        for (size_t j = 0 ; j < input_size ; ++j){     //100
            result[i] += weights[i][j] * input[j];
        }
        result[i] += bias[i];
    }

    return result;
}

Vector_double Dense::backward (const Vector_double& output_gradient
                                     , double learning_rate ) 
{
    Matrix_2d weight_grad (output_size , Vector_double(input_size , 0.00));
    Vector_double input_grad (input_size , 0.00);


    // Computing the gradient of the weights
    for (size_t i = 0; i < weight_grad.size(); ++i) {
    for (size_t j = 0; j < weight_grad[0].size(); ++j) {
        weight_grad[i][j] += output_gradient[i] * self_input[j]; // Element-wise multiplication
    }
   }


    //computing input gradient.
    Matrix_2d transpose_weights;
    transpose_weights.clear();
    transpose_weights = Transpose(weights);
    for (size_t i = 0 ; i < transpose_weights.size() ; ++i){
        for(size_t j= 0 ; j < transpose_weights[0].size() ; ++j){
            input_grad[i] += transpose_weights[i][j] * output_gradient[j];
        }
    }


    //updating weights - 2d matrix
    for (size_t i = 0 ; i < weights.size() ; ++i){
        for (size_t j = 0 ; j < weights[0].size() ; ++j){
            weights[i][j] -= learning_rate * weight_grad[i][j];
        }
    }

    
    //updating bias - 1D matrix
    for (size_t i = 0 ; i < bias.size() ; ++i){
        bias[i] -= learning_rate * output_gradient[i];
    }

    return input_grad;
}




