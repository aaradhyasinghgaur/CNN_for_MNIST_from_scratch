#include "data.h"

   data::data()
   {
      // feature_vector = new std::vector<unsigned int>;
   }

   data::~data()
   {
        
   }
   
   void data::set_feature_vector (Vector_double vect)
   {
      feature_vector = vect;
   }

   void data::append_to_feature_vector (uint8_t val)
   {
      feature_vector.push_back(val);
   }

   void data::set_label (unsigned int val)
   {
     label = val;
   }

   void data::set_enumerated_label (uint8_t val)
   {
      enum_label = val;
   }

   //getter functions.
   int data::get_feature_vector_size()
   {
     return feature_vector.size();
   }

   double data::get_label()
   {
      double temp = label;
      return temp;
   }
   int data::get_enumerated_label()
   {
      return enum_label;
   }

   Vector_double data::get_feature_vector ()
   {
      return feature_vector;
   }