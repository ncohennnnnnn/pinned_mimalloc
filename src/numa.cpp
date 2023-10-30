/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <hwmalloc/numa.hpp>
#include <hwmalloc/log.hpp>
#include <numaif.h>
#include <numa.h>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <sys/sysinfo.h>

#ifdef HWMALLOC_NUMA_THROWS
#include <stdexcept>
#define HWMALLOC_NUMA_ERROR(MSG) throw std::runtime_error("libnuma error: " MSG)
#else
#include <iostream>
#define HWMALLOC_NUMA_ERROR(MSG) std::cerr << "libnuma error: " << MSG << "\n"
#endif

#include <iostream>

namespace hwmalloc
{
bool                  numa_tools::is_initialized_ = false;
numa_tools::size_type numa_tools::page_size_ = sysconf(_SC_PAGESIZE);

namespace
{
bitmask* task_cpu_mask_ptr;
}

// construct the single instance
numa_tools::numa_tools() HWMALLOC_NUMA_CONDITIONAL_NOEXCEPT
{
    // initialize libnuma
    if (numa_available() < 0) HWMALLOC_NUMA_ERROR("could not initialize libnuma");
    else
    {
        task_cpu_mask_ptr = numa_allocate_cpumask();
        numa_sched_getaffinity(0, task_cpu_mask_ptr);
        discover_nodes();
    }
}

numa_tools::~numa_tools() noexcept { numa_free_cpumask(task_cpu_mask_ptr); }

// detect host and device nodes
void
numa_tools::discover_nodes() noexcept
{
    std::vector<index_type> host_nodes_;
    std::vector<index_type> local_nodes_;
    std::vector<index_type> device_nodes_;

    // detect each cpu's local node
    m_cpu_to_node.resize(get_nprocs());
    for (int cpu = 0; cpu < get_nprocs(); ++cpu)
    {
        m_cpu_to_node[cpu] = numa_node_of_cpu(cpu);
        if (numa_bitmask_isbitset(task_cpu_mask_ptr, cpu))
        {
            local_nodes_.push_back(m_cpu_to_node[cpu]);
        }
    }
    std::sort(local_nodes_.begin(), local_nodes_.end());
    local_nodes_.resize(
        std::unique(local_nodes_.begin(), local_nodes_.end()) - local_nodes_.begin());

    // allocate a cpu mask
    auto cpu_mask_ptr = numa_allocate_cpumask();
    // get maximum number of nodes
    const auto num_nodes = numa_all_nodes_ptr->size;
    // loop over nodes
    for (index_type i = 0; i < num_nodes; ++i)
    {
        // bit i is set if the calling process can access node i
        if (numa_bitmask_isbitset(numa_all_nodes_ptr, i))
        {
            // check whether the node is host or device by
            // converting node to a bitset of cpus associated with it
            numa_node_to_cpus(i, cpu_mask_ptr);

            // if there are any cpus, it is a host node
            if (numa_bitmask_weight(cpu_mask_ptr) > 0) host_nodes_.push_back(i);
            else
                device_nodes_.push_back(i);
        }
    }
    numa_free_cpumask(cpu_mask_ptr);

    m_host_nodes = node_map(std::move(host_nodes_));
    m_local_nodes = node_map(std::move(local_nodes_));
    m_device_nodes = node_map(std::move(device_nodes_));

    is_initialized_ = true;
}

numa_tools::index_type
numa_tools::local_node() const noexcept
{
    return m_cpu_to_node[sched_getcpu()];
}

bool
numa_tools::can_allocate_on(index_type node) const noexcept
{
    return local_nodes().count(node) > 0u;
}

numa_tools::allocation
numa_tools::allocate(size_type num_pages) const noexcept
{
    return allocate(num_pages, local_node());
}

numa_tools::allocation
numa_tools::allocate(size_type num_pages, index_type node) const noexcept
{
    if (num_pages == 0u) return {};
#ifndef HWMALLOC_NUMA_FOR_LOCAL
    // bypass numa allocation if on local node
    if (node == local_node()) return allocate_malloc(num_pages);
#endif
    // bypass numa allocation if node is not available
    if (!can_allocate_on(node)) return allocate_malloc(num_pages);
    auto ptr = numa_alloc_onnode(num_pages * page_size_, node);
    // fall back to malloc
    if (!ptr) return allocate_malloc(num_pages);
    HWMALLOC_LOG("allocating", num_pages * page_size_,
        "bytes using numa_alloc:", (std::uintptr_t)ptr);
    return {ptr, num_pages * page_size_, node};
}

numa_tools::allocation
numa_tools::allocate_malloc(size_type num_pages) const noexcept
{
    void* ptr = std::calloc(num_pages, page_size_);
    //void* ptr = std::malloc(num_pages * page_size_);
    if (!ptr) return {};
    HWMALLOC_LOG("allocating", num_pages * page_size_,
        "bytes using std::malloc:", (std::uintptr_t)ptr);
    return {ptr, num_pages * page_size_, get_node(ptr), false};
}

numa_tools::index_type
numa_tools::get_node(void* ptr) const noexcept
{
    int node_id = 0;
    get_mempolicy(&node_id, // mode: node id
        NULL,               // nodemask:  ignore
        0,                  // maxnode: ignore
        ptr,                // memory addr
        MPOL_F_ADDR         // return information about policy governing addr
            | MPOL_F_NODE   // return node id in mode
    );
    return static_cast<index_type>(node_id);
}

void
numa_tools::free(numa_tools::allocation const& a) const noexcept
{
    if (a)
    {
        if (a.use_numa_free)
        {
            HWMALLOC_LOG("freeing   ", a.size, "bytes using numa_free:", (std::uintptr_t)a.ptr);
            numa_free(a.ptr, a.size);
        }
        else
        {
            HWMALLOC_LOG("freeing   ", a.size, "bytes using std::free:", (std::uintptr_t)a.ptr);
            std::free(a.ptr);
        }
    }
}

// factory function
// only available from within this translation unit
numa_tools
make_numa_tools() HWMALLOC_NUMA_CONDITIONAL_NOEXCEPT
{
    return {};
}

// global instance of numa_tools with internal linkage
// only available within this translation unit
namespace
{
const numa_tools numa_tools_ = make_numa_tools();
}

// access to single instance
const numa_tools&
numa() noexcept
{
    return numa_tools_;
}

} // namespace hwmalloc
