#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <utility>
#include <random>
#include <unordered_map>
#include <chrono>
#include <CL/sycl.hpp>
using namespace std;
using namespace sycl;
typedef unsigned long long ull;
extern SYCL_EXTERNAL int randomInt(const int& l, const int& r, const ull& seed, const ull& offset);
extern SYCL_EXTERNAL void resloveConflict(vector<int>& visited, vector<int>& x, const vector<int>& y, const int& n, const int& l, const int& r);