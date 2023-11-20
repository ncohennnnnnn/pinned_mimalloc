#include <thread_affinity.hpp>
#include <scope_exit.hpp>

#include <hwloc.h>

#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
extern "C"
{
#include <sched.h>
#include <unistd.h>
}

namespace threading {

struct hwloc_error : std::runtime_error {
    hwloc_error(const std::string& msg, const std::string& err)
    : std::runtime_error(
          std::string{"Tried to use HWLOC and an operation failed with an error.\n"
                      "The problematic operation was: "} +
          msg + std::string{"\nIt returned this error:\n"} + err) {}
};

namespace {

inline void hwloc(int err, const std::string& msg) {
    if (0 != err) throw hwloc_error(msg, std::string{strerror(err)});
}

} // anonymous namespace

std::pair<int, int> hardware_resources() {
    // Create the topology and ensure we don't leak it
    auto topology = hwloc_topology_t{};
    auto topology_guard = on_scope_exit([&] { hwloc_topology_destroy(topology); });
    hwloc(hwloc_topology_init(&topology), "Topo init");
    hwloc(hwloc_topology_load(topology), "Topo load");
    //// Fetch our current restrictions and apply them to our topology
    //hwloc_cpuset_t cpus = hwloc_bitmap_alloc();
    //hwloc(cpus == NULL, "Bitmap allocation");
    //auto bitmap_guard = on_scope_exit([&] { hwloc_bitmap_free(cpus); });
    //hwloc(hwloc_get_cpubind(topology, cpus, HWLOC_CPUBIND_THREAD), "Get cpuset.");
    //hwloc(hwloc_topology_restrict(topology, cpus, 0), "Topo restrict.");
    // get the cores and PUs
    int cores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
    hwloc(cores <= 0, "Get CORE Objs");
    int proc = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
    hwloc(proc <= 0, "Get PROC Objs");
    return {cores, proc};
}

void
set_affinity(int tid, int count) {
    // Create the topology and ensure we don't leak it
    auto topology = hwloc_topology_t{};
    auto topology_guard = on_scope_exit([&] { hwloc_topology_destroy(topology); });
    hwloc(hwloc_topology_init(&topology), "Topo init");
    hwloc(hwloc_topology_load(topology), "Topo load");
    //// Fetch our current restrictions and apply them to our topology
    //hwloc_cpuset_t cpus = hwloc_bitmap_alloc();
    //hwloc(cpus == NULL, "Bitmap allocation");
    //auto bitmap_guard = on_scope_exit([&] { hwloc_bitmap_free(cpus); });
    //hwloc(hwloc_get_cpubind(topology, cpus, HWLOC_CPUBIND_THREAD), "Get cpuset.");
    //hwloc(hwloc_topology_restrict(topology, cpus, 0), "Topo restrict.");
    // Extract the root object describing the full local node
    auto root = hwloc_get_root_obj(topology);
    // Allocate one set per item
    auto cpusets = std::vector<hwloc_cpuset_t>(count, {});
    // Distribute items over topology, giving each of them as much private cache
    // as possible and keeping them locally in number order.
    hwloc(hwloc_distrib(topology, &root, 1,   // single root for the full machine
              cpusets.data(), cpusets.size(), // one cpuset for each thread
              INT_MAX,                        // maximum available level = Logical Cores
              0),                             // No flags
        "Distribute");
    // Bind threads to a single PU.
    hwloc(hwloc_bitmap_singlify(cpusets[tid]), "Singlify cpuset");
    // Now bind
    hwloc(hwloc_set_cpubind(topology, cpusets[tid], HWLOC_CPUBIND_THREAD), "Thread binding");
}

int
get_cpu() {
    return sched_getcpu();
}

} // namespace threading
