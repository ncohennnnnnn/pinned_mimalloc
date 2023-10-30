/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <vector>
#include <algorithm>

#ifdef HWMALLOC_NUMA_THROWS
#define HWMALLOC_NUMA_CONDITIONAL_NOEXCEPT
#else
#define HWMALLOC_NUMA_CONDITIONAL_NOEXCEPT noexcept
#endif

namespace hwmalloc
{
// Query numa memory regions and explicitely allocate on specific regions.
// This class is a thin wrapper over some of libnuma's functionality.
class numa_tools
{
  public:
    using index_type = std::size_t;
    using size_type = std::size_t;

    struct allocation
    {
        void* const      ptr = nullptr;
        size_type const  size = 0u;
        index_type const node = 0u;
        bool const       use_numa_free = true;

        operator bool() const noexcept { return (bool)ptr; }
    };

    // maps node id to index
    // where index is enumerating the nodes contiguously
    class node_map
    {
      public:
        using key_type = index_type;
        using mapped_type = index_type;
        using value_type = std::pair<const key_type, mapped_type>;

        node_map() = default;
        node_map(std::vector<index_type> nodes_)
        {
            if (!nodes_.empty())
            {
                std::sort(nodes_.begin(), nodes_.end());
                nodes_to_index.resize(nodes_.back() + 1, nodes_.size());
                nodes.reserve(nodes_.size());
                for (index_type idx = 0; idx < nodes_.size(); ++idx)
                {
                    nodes_to_index[nodes_[idx]] = idx;
                    nodes.push_back(value_type{nodes_[idx], idx});
                }
            }
        }
        node_map(node_map const&) = default;
        node_map(node_map&&) = default;
        node_map& operator=(node_map const&) = default;
        node_map& operator=(node_map&&) = default;

        size_type size() const noexcept { return nodes.size(); }
        auto      begin() const noexcept { return nodes.cbegin(); }
        auto      end() const noexcept { return nodes.cend(); }
        auto      rbegin() const noexcept { return nodes.crbegin(); }
        auto      rend() const noexcept { return nodes.crend(); }
        auto      find(key_type const& n) const noexcept
        {
            return (n >= upper_bound_key()) ? end() : (begin() + nodes_to_index[n]);
        }
        size_type count(key_type const& n) const noexcept
        {
            return (n >= upper_bound_key()) ? 0u : (nodes_to_index[n] < size());
        }

      private:
        index_type upper_bound_key() const noexcept { return nodes_to_index.size(); }

        std::vector<value_type> nodes;
        std::vector<index_type> nodes_to_index;
    };

  private:
    static bool      is_initialized_;
    static size_type page_size_;

  public:
    static bool      is_initialized() noexcept { return numa_tools::is_initialized_; }
    static size_type page_size() noexcept { return numa_tools::page_size_; }

  private:
    std::vector<index_type> m_cpu_to_node;
    node_map                m_host_nodes;
    node_map                m_local_nodes;
    node_map                m_device_nodes;

  private:
    numa_tools() HWMALLOC_NUMA_CONDITIONAL_NOEXCEPT;
    friend numa_tools make_numa_tools() HWMALLOC_NUMA_CONDITIONAL_NOEXCEPT;

  public:
    ~numa_tools() noexcept;

  public:
    const auto& host_nodes() const noexcept { return m_host_nodes; }
    const auto& local_nodes() const noexcept { return m_local_nodes; }
    const auto& device_nodes() const noexcept { return m_device_nodes; }
    index_type  local_node() const noexcept;

    bool       can_allocate_on(index_type node) const noexcept;
    allocation allocate(size_type num_pages) const noexcept;
    allocation allocate(size_type num_pages, index_type node) const noexcept;
    allocation allocate_malloc(size_type num_pages) const noexcept;
    void       free(allocation const& a) const noexcept;
    index_type get_node(void* ptr) const noexcept;

  private:
    void discover_nodes() noexcept;
};

const numa_tools& numa() noexcept;

} // namespace hwmalloc
