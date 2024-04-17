#ifndef BINARY_CROSS_ENTROPY_H
#define BINARY_CROSS_ENTROPY_H

#include "constants.h"

double mse (const Vector_double& y_true , const Vector_double& y_pred)
{
   if (y_true.size() != y_pred.size())
   {
    std::cout << "size of y_true : " << y_true.size() << std::endl;
    std::cout << "size of y_pred : " << y_pred.size() << std::endl;
    for (int i = 0 ; i < 10 ; ++i){
        std::cout << "printing predicted otuput :" << y_pred[i] << std::endl;
    }

    std::cout << "printing true otuput :" << y_true[0] << std::endl; 
    std::cerr << "Error : size of predicted values and actual values isn't same." << std::endl;
    exit(0);
   }

    double square_diff_sum = 0.00 ;
    for (size_t i = 0 ; i < y_pred.size() ; ++i){
        double difference = y_pred[i] - y_true[i];
        square_diff_sum = difference * difference;
    }
    return square_diff_sum / y_pred.size();
}

Vector_double mse_prime (const Vector_double& y_true , const Vector_double& y_pred)
{
    if (y_true.size() != y_pred.size())
   {
    std::cerr << "Error : size of predicted values and actual values isn't same." << std::endl;
    exit(0);
   }
    
    Vector_double output_gradient (y_true.size() , 0.00);
    for (size_t i = 0; i < y_true.size() ; ++i){
        output_gradient[i] = 2 * (y_pred[i] - y_true[i] ) / y_true.size();
    }

    return output_gradient;
}

double binary_cross_entropy(const Vector_double& y_true , const Vector_double& y_pred)
{
   if (y_true.size() != y_pred.size())
   {
    std::cout << "size of y_true : " << y_true.size() << std::endl;
    std::cout << "size of y_pred : " << y_pred.size() << std::endl;
    std::cerr << "Error : size of predicted values and actual values isn't same." << std::endl;
    exit(0);
   }

    double diff_sum = 0.00 ;
    for (unsigned i = 0 ; i < y_pred.size() ; ++i){
        double difference = (-y_true[i] * log(y_pred[i])) - ((1 - y_true[i]) * log( 1 - y_pred[i])) ;
        diff_sum += difference ;
    }

    return diff_sum / y_pred.size();
}

Vector_double binary_cross_entropy_prime (const Vector_double& y_true , const Vector_double& y_pred)
{
    if (y_true.size() != y_pred.size())
   {
    std::cerr << "Error : size of predicted values and actual values isn't same." << std::endl;
    exit(0);
   }
    
    Vector_double output_gradient (y_true.size() , 0.00);

    for (size_t i = 0; i < y_true.size() ; ++i){
        output_gradient[i] = ( (1 - y_true[i] / 1 - y_pred[i]) - (y_true[i] - y_pred[i]))  / y_true.size() ;
    }

    return output_gradient;
}

double categorical_cross_entropy(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Error: size of predicted values and actual values are not the same.");
    }

    double loss = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        loss += y_true[i] * std::log(y_pred[i] + 1e-9);
    }

    return -loss;
}

Vector_double categorical_cross_entropy_prime(const Vector_double& y_true, const Vector_double& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Error: size of predicted values and actual values are not the same.");
    }

    Vector_double gradient(y_pred.size(), 0.0);
    for (size_t i = 0; i < y_true.size(); ++i) {
        gradient[i] = -(y_true[i] / (y_pred[i] + 1e-9));  // Add small epsilon to avoid division by zero
    }

    return gradient;
}


#endif
