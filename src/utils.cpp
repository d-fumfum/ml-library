#include "ml/utils.h"

#include <cmath>
#include <cstring>
#include <iostream>

int error_occured(const std::string& error){
    std::cerr << "Error occured: " << error << '\n';
    return -1;
}

void* my_memcpy(void* dst, const void* src, std::size_t n) noexcept {
    if (n == 0 || dst == src) return dst;
    return std::memcpy(dst, src, n);
}

float my_exp(float x) noexcept {
    return std::exp(x);
}


float my_log(float x) noexcept {
    return std::log(x);
}


float my_max(float a, float b) noexcept{
    return std::fmax(a, b);
}

float my_sqrt(float x) noexcept {
    return std::sqrt(x);
}
