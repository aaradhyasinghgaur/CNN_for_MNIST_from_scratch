#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "layers.h"

class Activation : public Layer
{
public:
    Activation () = default;
    Activation (std::function<double(double)> activation , std::function<double(double)> activation_prime)
    {
          this -> activation = activation;
          this -> activation_prime = activation_prime;
    }

    virtual ~Activation() = default;
    virtual Vector_double forward (const Vector_double& input) override ;
    virtual Vector_double backward (const Vector_double& output_gradient
                            , double learning_rate ) override ;

private:
Vector_double activation_input ;
std::function <double(double)> activation;
std::function <double(double)> activation_prime;
};

#endif