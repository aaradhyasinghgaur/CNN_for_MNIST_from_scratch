#include "activation.h"

Vector_double Activation::forward (const Vector_double& input)
{
    activation_input.clear();
    activation_input = input;
   Vector_double result (input.size() , 0.00);
    for (size_t i = 0 ; i < input.size() ; ++i)
    {
        result[i] = activation(input[i]);
    }

    return result;
  
}

Vector_double Activation::backward (const Vector_double& output_gradient , double learning_rate )
{
    Vector_double input_grad (output_gradient.size() , 0.00);
    for(size_t i = 0; i < output_gradient.size() ; ++i){
        input_grad[i] = output_gradient[i] * activation_prime(activation_input[i]);
    }
    
    return input_grad;
}