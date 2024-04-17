#include "network.h"
#include "loss.h"
#include "matrix.h"

Network::Network()
{
    c1 = new Convolution({1, 28, 28}, {5, 1, 3, 3});
    s1 = new Sigmoid();
    d1 = new Dense(5 * 26 * 26, 100);
    s2 = new Sigmoid();
    d2 = new Dense(100, 10);
    s3 = new Softmax();
    layers = {c1 , s1, d1, s2, d2, s3} ;
}

int Network::argmax (const Vector_double& vec) {
    return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
}


float Network::accuracy_metric(Matrix_2d actual , Matrix_2d predicted)
{
   int correct = 0;
   for (unsigned int i = 0 ; i < actual.size() ; ++i){
    int actual_class_index = argmax(actual[i]);
    int predicted_class_index = argmax(predicted[i]);

    if(actual_class_index == predicted_class_index){
        correct += 1;
    }
   }
   return static_cast<float>(correct) / actual.size() * 100.0;
}
 


void Network::read_training_dataset()
{
    dh = new data_handler();
    dh->read_feature_vector("train-images.idx3-ubyte");
    dh->read_feature_label("train-labels.idx1-ubyte");
    dh->split_data();
    dh->count_classes();
}

void Network::read_test_dataset()
{
    dh = new data_handler();
    dh->read_feature_vector("t10k-images.idx3-ubyte");
    dh->read_feature_label("t10k-labels.idx1-ubyte");
    dh->count_classes();
}


void Network::training(double epochs, double learning_rate ) {
    std::vector<data*>* ptr_train = dh->get_training_data();
    int size_training_data = ptr_train->size();                    //training data = 54,000

    std::cout << "Total size of training data: " << size_training_data << std::endl;

   int j;
   int count = 1;
   for ( j = 0; j < size_training_data ; j = j + 2000)     //for each mini batch = 2000
   {
    std::cout << "mini batch = " << count << std::endl;
    for (int e = 0; e < epochs ; ++e) {              //each epoch = 10
        double error = 0.0;
        
        int i;
        for (i = j ; i < j + 2000 ; ++i) {                                //each image in mini bach.
            Vector_double x = ptr_train->at(i)->get_feature_vector();
            double real_label = ptr_train->at(i)->get_label();

            Vector_double output = x;
            for (auto* layer : layers) {
                output = layer->forward(output);
            }


            Vector_double y(10, 0);
            y[real_label] = 1;

            error += categorical_cross_entropy(y, output);
            Vector_double grad(10 , 0);
            grad =  categorical_cross_entropy_prime(y, output);


            Layer* prev_layer = nullptr;
            for (auto it = std::rbegin(layers); it != std::rend(layers); ++it) {
                Layer* layer = *it;
                grad = layer->backward(grad, learning_rate );
                prev_layer = layer;
            }

        }

        error /= 2000;
        std::cout << "Epoch: " << e << " Average Error: " << error << std::endl;


    }
    count++;
    std::cout << "Training Done for : " << j  << " to " << j + 2000 <<  " data. " << std::endl;
    validation();
   }
}


void Network::validation()
{     
    std::vector<data*>* ptr_validation = dh->get_validation_data();                  
    int size_validation_data = ptr_validation->size() ;                           // validation data = 6,000 
    
    Matrix_2d predicted (size_validation_data , Vector_double(10 , 0));                     
    Matrix_2d actual (size_validation_data , Vector_double(10 , 0));                                

    for (int i = 0 ; i < size_validation_data ; ++i) {
        Vector_double x = ptr_validation->at(i)->get_feature_vector();  
        double real_label = ptr_validation->at(i)->get_label();   
        //forward

        Vector_double output = x ;
        for (auto* layer : layers ) {
            output = layer -> forward(output);
        }

        Vector_double y(10, 0);
        y[real_label] = 1;


        if (output.size() == y.size()) {
            for (unsigned j = 0 ; j < output.size() ; ++j) {
            predicted[i][j] = output[j];
            actual[i][j] = y[j];
            }
        }

    }

  std::cout << "  Validation set : Classification Accuracy (percentage) = " << accuracy_metric(actual , predicted) << std::endl;

}

void Network::test()
{     
    std::vector<data*>* ptr_test = dh->get_data_array();                  
    int size_test_data = ptr_test->size() ;                           // test data = 10,000
    
    Matrix_2d predicted (size_test_data , Vector_double(10 , 0));                     
    Matrix_2d actual (size_test_data , Vector_double(10 , 0));                                

    for (int i = 0 ; i < size_test_data ; ++i) {
        Vector_double x = ptr_test->at(i)->get_feature_vector();  
        double real_label = ptr_test->at(i)->get_label();   
        //forward

        Vector_double output = x ;
        for (auto* layer : layers ) {
            output = layer -> forward(output);
        }

        Vector_double y(10, 0);
        y[real_label] = 1;


        if (output.size() == y.size()) {
            for (unsigned j = 0 ; j < output.size() ; ++j) {
            predicted[i][j] = output[j];
            actual[i][j] = y[j];
            }
        }

    }

  std::cout << " Test set : Classification Accuracy (percentage) = " << accuracy_metric(actual , predicted) << std::endl;

}


