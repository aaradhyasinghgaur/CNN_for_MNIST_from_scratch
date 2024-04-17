#include "network.h"                                  

#include <iostream>
#include <chrono>

int main() {

    auto start = std::chrono::high_resolution_clock::now();

    double epochs = 0;
    double learning_rate = 0;
 
    Network *net ;
    net = new Network();
    net->read_training_dataset();
   
    net->training(epochs = 10 , learning_rate = 0.001 );
    net->read_test_dataset();
    net->test();

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;

    return 0;
}

