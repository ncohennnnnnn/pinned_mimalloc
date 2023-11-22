#include <allocator.hpp>
#include <task_group.hpp>

#include <barrier>
#include <iostream>
#include <math.h>

/* TODO:
    - single arena with big size or one arena per thread ?
    - numa node stuff, steal it from Fabian and get how to use it
    - for now, MI_OVERRIDE has to be set to OFF otherwise we can't use
   std::aligned_alloc, try to find a way to either write ur own aligned_alloc or
   keep it like this.
    - device stuff
        - get the device id in the device_memory constructor
        - check if ptr is actually on device for user_device_memory
    - RMA keys functions, the ones for individual objects (with offset)
    - UCX
    - MPI
    - concepts for Key, Register and Malloc
    - choose appropriate flags in the cuda pinning
    - choose appropriate flags in the libfabric backend methods
    - add the choice of which numa node one wants to allocate using std::malloc
    - set is_large as an option in the ex_mimalloc constructor (and hence in
   resource and resource_builder)
    - (the rest of the allocator class)
*/

// void mi_heap_destroy(mi_heap_t* heap);
using unique_tls_heap = std::unique_ptr<mi_heap_t, void (*)(mi_heap_t *)>;
thread_local unique_tls_heap thread_local_heap_{nullptr, mi_heap_destroy};

mi_heap_t *create_tls_heap(mi_arena_id_t m_arena_id) {
  return mi_heap_new_in_arena(m_arena_id);
}

void heap_per_thread(std::size_t mem);

template <typename Alloc>
void fill_array_multithread(const int nb_threads, const int nb_allocs, Alloc a);

template <typename Alloc> void std_vector(Alloc a);

template <typename Alloc> void fill_buffer(Alloc a);

template <typename Alloc> void usual_alloc(Alloc a);

struct thing {
  thing() { std::cout << "constructing" << std::endl; }
  ~thing() { std::cout << "destructing" << std::endl; }
};

int main() {
  std::size_t mem = 1ull << 30;

  /* Build resource and allocator by hand */
  // using resource_t = resource <context <pinned <host_memory <base>> ,
  // backend> , ex_mimalloc> ; using alloc_t = pmimallocator<int, resource_t>;
  // auto res = std::make_shared<resource_t>(mem);
  // alloc_t a(res);
  // fmt::print("\n");

  /* Build resource and allocator via resource_builder */

  //  resource_builder RB;
  //  auto rb = RB.use_mimalloc().pin().register_memory().on_host(mem);
  //  using resource_t = decltype(rb.build());
  //  using alloc_t = pmimallocator<int, resource_t>;
  //  alloc_t a(rb);
  //  fmt::print("\n\n");

  /* Build an arena, then 1 heap per thread, allocate with them */
  heap_per_thread(mem);

  /* Fill an array through several threads and deallocate all on thread 0*/
  // fill_array_multithread(2, 5, a);

  /* Standard vector (doesn't work) */
  // std_vector(a);

  /* Buffer filling */
  // fill_buffer(a);

  /* Usual allocation */
  // usual_alloc(a);
}

/* Fill an array through several threads and deallocate all on thread 0*/
template <typename Alloc>
void fill_array_multithread(const int nb_threads, const int nb_allocs,
                            Alloc a) {
  int *p[nb_threads * nb_allocs];
  std::barrier sync_point(nb_threads);
  threading::task_system ts(nb_threads, true);
  threading::parallel_for::apply(
      nb_threads, &ts, [a, &p, &sync_point, &nb_allocs](int thread_id) mutable {
        fmt::print("Thread {} \n", thread_id);
        int idx;
        for (std::size_t i = 0; i < nb_allocs; ++i) {
          idx = nb_allocs * thread_id + i;
          p[idx] = a.allocate(2);
          fmt::print("{} : ptr allocated \n", static_cast<void *>(p[idx]));
          *p[idx] = idx;
        }
        sync_point.arrive_and_wait();
      });
  for (int i = 0; i < nb_allocs - 1; ++i) {
    if (p[i + 1] - p[i] != 1) {
      a.deallocate(p[i]);
    } else {
      fmt::print("[ERROR] from thread {} \n", floor(i / nb_allocs));
    }
  }
  a.deallocate(p[nb_allocs]);
  fmt::print("\n\n");
  mi_collect(true);
  mi_stats_print(NULL);
}

/* Standard vector */
template <typename Alloc> void std_vector(Alloc a) {
  fmt::print("Standard vector \n");
  // fmt::print("Resource use count : {} \n", res.use_count());
  std::vector<int, Alloc> v(100, a);
  // fmt::print("Resource use count : {} \n", res.use_count());
  fmt::print("{} : Vector data \n", (void *)v.data());
  for (std::size_t i; i < 100; ++i) {
    v[i] = 1;
  }
  for (std::size_t i; i < 100; ++i) {
    fmt::print("{}, ", v[i]);
  }
  fmt::print("\n\n");
}

/* Buffer filling */
template <typename Alloc> void fill_buffer(Alloc a) {
  fmt::print("Buffer filling\n");
  int *buffer[1000];
  for (std::size_t i; i < 100; ++i) {
    buffer[i] = a.allocate(8);
  }
  for (std::size_t i; i < 100; ++i) {
    a.deallocate(buffer[i]);
  }
  fmt::print("\n\n");
}

/* Usual allocation */
template <typename Alloc> void usual_alloc(Alloc a) {
  fmt::print("Usual allocation\n");
  int *p1 = a.allocate(32);
  int *p2 = a.allocate(48);
  a.deallocate(p1);
  a.deallocate(p2);
  fmt::print("\n\n");
}

/* Build an arena, then 1 heap per thread, allocate with them */
void heap_per_thread(std::size_t mem) {
  mi_option_set(mi_option_limit_os_alloc, 1);
  host_memory<base> hm(mem);
  void *ptr = hm.get_address();
  constexpr std::size_t nb_threads = 20;
  constexpr std::size_t nb_allocs = 100000;
  mi_arena_id_t m_arena_id{};

  mi_heap_t *heaps[nb_threads];

  std::vector<uint32_t *> ptrs(nb_threads * nb_allocs);
  bool success = mi_manage_os_memory_ex(ptr, mem, true, false, true, -1, true,
                                        &m_arena_id);
  if (!success) {
    fmt::print("{} : [error] ex_mimalloc failed to create the arena. \n", ptr);
  } else {
    fmt::print("{} : Mimalloc arena created \n", ptr);
  }
  fmt::print("\n");

  /*
    threading::task_system ts(nb_threads, true);
    threading::parallel_for::apply(
        nb_threads, &ts,
        [&heaps, m_arena_id, &nb_allocs,
         &ptrs ](int thread_id) mutable {
          std::cout << std::this_thread::get_id() << std::endl;

          heaps[thread_id] = mi_heap_new_in_arena(m_arena_id);
          for (int i = 0; i < nb_allocs; ++i) {
            ptrs[thread_id * nb_allocs + i] =
                static_cast<uint32_t *>(mi_heap_malloc(
                    heaps[thread_id], 32));
            *ptrs[thread_id * nb_allocs + i] = thread_id * nb_allocs + i;
          }
        });
  */

  std::vector<std::thread> threads;
  for (std::size_t thread_id = 0; thread_id < nb_threads; ++thread_id) {
    threads.push_back(std::thread{
        [&heaps, m_arena_id, &nb_allocs,
         &ptrs /*, &sync_point*/](int thread_id) mutable {
          std::cout << thread_id << ": " << std::this_thread::get_id()
                    << std::endl;
          // std::cout << _mi_thread_id() << std::endl;
          // heaps[thread_id] = mi_heap_new_in_arena(m_arena_id);
          if (!thread_local_heap_) {
            std::cout << "New heap on thread " << thread_id << std::endl;
            auto my_delete = [](mi_heap_t *heap) {
              std::cout << "NOT Deleting heap (it's safe) " << heap
                        << std::endl;
              // mi_heap_destroy(heap);
            };
            thread_local_heap_ =
                unique_tls_heap{mi_heap_new_in_arena(m_arena_id), my_delete};
          }

          for (int i = 0; i < nb_allocs; ++i) {
            ptrs[thread_id * nb_allocs + i] =
                static_cast<uint32_t *>(mi_heap_malloc(
                    /*heaps[thread_id]*/ thread_local_heap_.get(), 32));
            *ptrs[thread_id * nb_allocs + i] = thread_id * nb_allocs + i;
          }
        },
        thread_id});
  }
  for (auto &t : threads)
    t.join();
  std::cout << "finished" << std::endl;

  std::cout << "Clearing memory " << std::endl;

  for (int i = 0; i < nb_allocs * nb_threads; ++i) {
    int thread_id = i / nb_allocs;
    // fmt::print("{} \n", thread_id);
    if (*ptrs[i] == i) {
      mi_free(ptrs[i]);
    } else {
      fmt::print("[ERROR] from thread {}, expected {}, got {} \n", thread_id, i,
                 *ptrs[i]);
    }
  }

  threads.clear();

  for (int i = 0; i < nb_threads; ++i) {
    // mi_heap_destroy(heaps[i]);
  }

  fmt::print("\n\n");
  mi_collect(true);
  mi_stats_print(NULL);
}
