//functions - reading data and storing it in data structure.

#ifndef DATA_H
#define DATA_H

#include "constants.h"

class data
{
public:
  //setter functions
   data();
   ~data();
   void set_feature_vector (Vector_double );
   void append_to_feature_vector (uint8_t);
   void set_label (unsigned int);
   void set_enumerated_label (uint8_t);


   //getter functions.
   int get_feature_vector_size();
   double get_label();
   int get_enumerated_label();
   Vector_double get_feature_vector ();

private:
Vector_double feature_vector; //no class at end.
double label;
int enum_label; // A -> 1 , B -> 2


};

#endif