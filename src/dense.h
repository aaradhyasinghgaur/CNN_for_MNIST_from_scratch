#ifndef DENSE_H
#define DENSE_H

#include "layers.h"


class Dense : public Layer
{
public:
    Dense() = default;
    ~Dense() = default;
    Dense (int input_size , int output_size);
    virtual Vector_double  forward (const Vector_double & input_) override;
    virtual Vector_double  backward(const  Vector_double & output_gradient
                            , double learning_rate ) override;

protected:
    int output_size = 0;
    int input_size = 0;

    Vector_double self_input;
    Matrix_2d weights;
    Vector_double bias;

};


#endif