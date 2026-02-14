#pragma once

#include <cstddef>
#include <string>

inline constexpr float euler = 2.71828182845904523536f;

int error_occured(const std::string& error);
float my_exp(float x) noexcept;
float my_log(float x) noexcept;
float my_max(float a, float b) noexcept;
float my_sqrt(float x) noexcept;
void* my_memcpy(void* dst, const void* src, std::size_t n) noexcept;