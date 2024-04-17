#ifndef ACTIVATION_FUNTIONS_H 
#define ACTIVATION_FUNCTIONS_H

#include "activation.h"
#include "layers.h"
#include "constants.h"

class Tanh : public Activation 
{
public:
   Tanh() : Activation (
      [](double x) {return std::tanh(x);},
      [](double x) {double tanhx = std::tanh(x);
                    return 1.0 - tanhx * tanhx ;}
   ) {}

   virtual ~Tanh() = default;
} ;

class Sigmoid : public Activation 
{
public:
   Sigmoid() : Activation (
      [](double x) { return 1 / (1 + exp(-x));},
      [](double x) { return  exp(-x) / pow ( (1 + exp(-x)) , 2 ); }
   ) {}

   virtual ~Sigmoid() = default;
};


class ReLu: public Activation 
{
public:
   ReLu() : Activation (
      [](double x) { return std::max(0.0 , x); },
      [](double x) { return (x >= 0) ? 1 : 0 ;  }  
   ) {}

   virtual ~ReLu() = default;
};


class Softmax : public Layer
{
public:
   Softmax() = default;
   ~Softmax() = default;
   Vector_double forward (const Vector_double& input_) 
   {
     double sum_exp = 0.0;

    // Compute the sum of exponentials of input.
    for (unsigned int i = 0; i < input_.size() ; ++i) {
        sum_exp += exp(input_[i]);
    }

    // Compute softmax probabilities
    self_output.clear();
    for (unsigned int i = 0; i < input_.size() ; ++i) {
        self_output.push_back(exp(input_[i]) / sum_exp);
    }

    return self_output;
   }


   Vector_double backward (const Vector_double& output_gradient , double learning_rate )  
   {
      Vector_double grad(self_output.size() , 0.00);
   
      for (size_t i = 0; i < self_output.size(); i++) {
        double activation_grad = 0.0;
        for (size_t j = 0; j < self_output.size(); j++) {
          if(i == j)
          {
            activation_grad += self_output[i] * (1 - self_output[i]) * output_gradient[j];
          }
          else 
          {
            activation_grad += (-self_output[i]) * self_output[j] * output_gradient[j];
          }
        }
        grad[i] = activation_grad ;
    }

    return grad;
   }

private:
Vector_double self_output ;
};


#endif