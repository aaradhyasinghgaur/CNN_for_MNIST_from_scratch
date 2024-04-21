#ifndef MATRIX_H
#define MATRIX_H

#include "constants.h"

double Generate_Random_Normal();
Matrix_2d Transpose (Matrix_2d matrix) ;
Vector_double Generate_Random_Vector (int dim1) ;
Matrix_4d rotate_matrix_180(const Matrix_4d& matrix) ;
Matrix_3d padding_matrix (Matrix_3d matrix , int padding) ;
Matrix_3d set_random_matrix_3d(Vector_int output_dim) ;
Matrix_3d reshape_matrix_to_3d (const Vector_int& dim , const Vector_double &input);
Vector_double reshape_matrix_to_1d (Vector_int dim , Matrix_3d input);
Matrix_2d initializeWeights_2d(int , int);
Matrix_4d initializeWeights(Vector_int );



#endif
