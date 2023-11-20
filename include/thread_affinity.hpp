#pragma once

#include <utility>

namespace threading {

// returns available resources for the current process: [physical cores, logical processors]
std::pair<int, int> hardware_resources();

// sets the affinity of a thread with index `tid` within a group of `count` threads
void set_affinity(int tid, int count);

// get the current logical processor index
int get_cpu();

} // namespace threading
