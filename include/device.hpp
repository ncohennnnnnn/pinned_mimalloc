#pragma once

#include <cstddef>


void* device_allocate(std::size_t size);

// void* device_reallocate(void* ptr, std::size_t size);

void device_deallocate(void* ptr) noexcept;

int get_num_devices();

int get_device_id();

void set_device_id(int id);

void memcpy_to_device(void* dst, void const* src, std::size_t count);

void memcpy_to_host(void* dst, void const* src, std::size_t count);
