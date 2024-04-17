#include "matrix.h"

double Generate_Random_Normal() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<double> distribution(0.0, 1.0);
    return distribution(gen);
}

Matrix_2d Transpose (Matrix_2d matrix){
    // Create a new matrix to store the transpose
    Matrix_2d transposedMatrix (matrix[0].size(), Vector_double(matrix.size()));

    // Perform the transpose operation
    for (unsigned i = 0; i < matrix.size(); i++) {
        for (unsigned int j = 0; j < matrix[0].size(); j++) {
            transposedMatrix[j][i] = matrix[i][j];
        }
    }
    return transposedMatrix;
}

Vector_double Generate_Random_Vector (int dim1)
{
   Vector_double temp (dim1 , 0.00);
    for (unsigned i = 0 ; i < dim1 ; ++i)
    {
        temp[i] = Generate_Random_Normal ();
    }
    return temp;
}

Matrix_2d Generate_Random_Matrix (int dim1 , int dim2)
{
    Matrix_2d temp (dim1 , Vector_double(dim2, 0.00));
    for (unsigned i = 0 ; i < dim1 ; ++i) {
      for (unsigned j = 0 ; j < dim2 ; ++j) {
        temp[i][j] = Generate_Random_Normal();
      }
    }
    return temp;
}



Matrix_4d rotate_matrix_180(const Matrix_4d& matrix) {
    int no_of_kernels = matrix.size();
    int depth = matrix[0].size();
    int rows = matrix[0][0].size();
    int cols = matrix[0][0][0].size();

    Matrix_4d temp(no_of_kernels, Matrix_3d(depth, Matrix_2d(rows, Vector_double(cols, 0))));

    for (int k = 0; k < no_of_kernels; ++k) {
        for (int j = 0; j < depth; ++j) {
            for (int i = 0; i < rows; ++i) {
                temp[k][j][i] = Vector_double(matrix[k][j][i].rbegin(), matrix[k][j][i].rend());
            }
            reverse(temp[k][j].begin(), temp[k][j].end());
        }
    }

   return temp;
}

Matrix_3d padding_matrix (Matrix_3d matrix , int padding) {
    int depth = matrix.size();      //5
    int rows = matrix[0].size() ;   //26
    int cols = matrix[0][0].size() ;    //26

                    //5                  //28                                     //28
     Matrix_3d temp(depth, Matrix_2d(rows + (2 * padding) , Vector_double(cols + (2 * padding), 0.0)));

        for (int j = 0; j < depth ; ++j) {
            for (int k = 0 ; k < rows ; ++k){
               for (int l = 0 ; l < cols ; ++l){
                  temp[j][k + padding][l + padding] = matrix[j][k][l];
               }
            }
        }

   return temp;
}


Matrix_4d set_random_matrix_4d(Vector_int kernel_dim)
{
   Matrix_4d temp(kernel_dim[0],Matrix_3d( kernel_dim[1],Matrix_2d( kernel_dim[2], Vector_double(kernel_dim[3] , 0) )));    

    for (int i = 0 ; i < kernel_dim[0] ; ++i) {                   //5
      for (int j = 0 ; j < kernel_dim[1] ; ++j) {          // 1
         for (int k = 0 ; k < kernel_dim[2] ; ++k)  {             // 3
            for (int m = 0 ; m < kernel_dim[3] ; ++m)  {                // 3
               temp[i][j][k][m] = Generate_Random_Normal();  }  }  }  }
    return temp;
}


Matrix_3d set_random_matrix_3d(Vector_int output_dim)
{
    Matrix_3d temp( output_dim[0], Matrix_2d( output_dim[1], Vector_double(output_dim[2] , 0) ) );


    for (int i = 0 ; i < output_dim[0] ; ++i) {   
      for (int j = 0 ; j < output_dim[1] ; ++j) {   
         for (int k = 0 ; k < output_dim[2] ; ++k)  {   
            temp[i][j][k] = Generate_Random_Normal();   }  }  }
    return temp;
}


Matrix_3d reshape_matrix_to_3d (const Vector_int& dim , const Vector_double &input)
{
   Matrix_3d temp( dim[0], Matrix_2d( dim[1], Vector_double(dim[2] , 0.0) ));

   for (int i = 0 ; i < dim[0] ; ++i) {   
      for (int j = 0 ; j < dim[1] ; ++j) {   
         for (int k = 0 ; k < dim[2] ; ++k)  {   
            temp[i][j][k] = input[i * dim[1] * dim[2] + j * dim[2] + k];   }  }  }
    return temp;
}


Vector_double reshape_matrix_to_1d (Vector_int dim , Matrix_3d input)
{
    Vector_double temp( dim[0] * dim[1] * dim[2] , 0.0);


   for (int i = 0 ; i < dim[0] ; ++i) {   
      for (int j = 0 ; j < dim[1] ; ++j) {   
         for (int k = 0 ; k < dim[2] ; ++k)  {   
                int index = i * dim[1] * dim[2] + j * dim[2] + k;
                temp[index] = input[i][j][k];  }  }  }
    
    return temp;
}



    Matrix_4d initializeWeights(Vector_int kernel_dim) {

        std::random_device rd;
        std::mt19937 gen(rd());

        Matrix_4d temp;
        double std_dev = sqrt(2.0 / (kernel_dim[1] * kernel_dim[2] * kernel_dim[3] + kernel_dim[0]));

        std::normal_distribution<double> distribution(0, std_dev);

        temp.clear();

        temp.resize(kernel_dim[0]);
        for (int i = 0; i < kernel_dim[0]; ++i) {
             temp[i].resize(kernel_dim[1]);
            for (int j = 0; j < kernel_dim[1]; ++j) {
                temp[i][j].resize(kernel_dim[2]);
                for (int k = 0; k < kernel_dim[2]; ++k) {
                   temp[i][j][k].resize(kernel_dim[3]);
                    for (int l = 0; l < kernel_dim[3]; ++l) {
                      temp[i][j][k][l] = distribution(gen);
                    }
                }
            }
        }
        return temp;
    }

    Matrix_2d initializeWeights_2d(int outputSize, int inputSize) {
    std::random_device rd;
    std::mt19937 gen(rd());

    Matrix_2d temp (outputSize , Vector_double(inputSize, 0.00));
    double std_dev = sqrt(2.0 / (inputSize + outputSize));

    std::normal_distribution<double> distribution(0, std_dev);

    for (int i = 0; i < outputSize; ++i) {
    for (int j = 0; j < inputSize ; ++j) {
        temp[i][j] = distribution(gen);
    }
    }

    return temp;
}

