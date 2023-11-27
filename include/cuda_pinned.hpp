#pragma once

#include <cuda_runtime.h>
#include <fmt/core.h>

template <typename Memory>
/** @brief CUDA-pinned memory living on the host. */
class cuda_pinned : public Memory
{
public:
    cuda_pinned()
      : Memory{}
      , m_pinned{false}
    {
    }

    cuda_pinned(Memory&& mem)
      : Memory{std::move(mem)}
      , m_pinned{false}
    {
        _pin_or_unpin(Memory::m_address, Memory::m_size, true);
    }

    cuda_pinned(const std::size_t size, const std::size_t alignement = 0)
      : Memory{size, alignement}
      , m_pinned{false}
    {
        _pin_or_unpin(Memory::m_address, Memory::m_size, true);
    }

    cuda_pinned(void* ptr, const std::size_t size)
      : Memory{ptr, size}
      , m_pinned{false}
    {
        _pin_or_unpin(Memory::m_address, Memory::m_size, true);
    }

    cuda_pinned(cuda_pinned&&) noexcept = default;

    ~cuda_pinned()
    {
        if (m_pinned)
        {
            _pin_or_unpin(Memory::m_address, Memory::m_size, false);
        }
    }

private:
    void _pin_or_unpin(void* ptr, const size_t size, bool pin)
    {
        cudaError_t cudaStatus;
        if (pin)
        {
            /* TODO: Choose the appropriate flags */
            cudaStatus = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
            if (cudaStatus != cudaSuccess)
            {
                fmt::print("cudaHostRegister failed: {} \n", cudaGetErrorString(cudaStatus));
                m_pinned = false;
            }
            else
            {
                fmt::print("{} : Memory pinned (by CUDA). \n", ptr);
                m_pinned = true;
            }
        }
        else
        {
            /* TODO: Choose the appropriate flags */
            cudaStatus = cudaHostUnregister(ptr);
            if (cudaStatus != cudaSuccess)
            {
                fmt::print("cudaHostUnregister failed: {} \n", cudaGetErrorString(cudaStatus));
            }
            else
            {
                fmt::print("{} : Memory unpinned (by CUDA). \n", ptr);
                m_pinned = false;
            }
        }
    }

protected:
    bool m_pinned;
};