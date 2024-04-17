#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <iostream>
#include <cstdint>         //uint8_t
#include <vector>
#include <unordered_set>   // to keep track of indeexes when we split data.
#include <fstream>        //to raed file I/O
#include <random>
#include <bitset>
#include <map>      // map  class label to enumerated value.
#include <cmath>
#include <functional>
#include <memory>
#include <cstdlib>
#include <algorithm>
#include <ctime>
#include <string>
#include <stddef.h>

typedef std::vector<int> Vector_int ;
typedef std::vector<double> Vector_double ;
typedef std::vector<Vector_double> Matrix_2d ;
typedef std::vector<Matrix_2d> Matrix_3d ;
typedef std::vector<Matrix_3d> Matrix_4d ;

#endif