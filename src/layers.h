#ifndef LAYER_H
#define LAYER_H

#include "constants.h"

//abstract class
class Layer 
{
public:
   Layer () = default;
   virtual ~Layer () = default;

   //pure virtual functions
   virtual Vector_double forward (const Vector_double& input_) = 0;
   virtual Vector_double backward (const Vector_double& output_gradient
                            , double learning_rate)  = 0;



};

#endif