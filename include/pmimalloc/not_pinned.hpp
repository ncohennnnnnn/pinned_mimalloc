template <class Memory>
/** @brief Host or device memory that is not to be pinned. */
class not_pinned : public Memory
{
public:
    not_pinned()
      : Memory{}
    {
    }

    not_pinned(Memory&& mem)
      : Memory{std::move(mem)}
    {
    }

    not_pinned(const std::size_t size, const std::size_t alignment = 0)
      : Memory{size, alignment}
    {
    }

    not_pinned(void* ptr, const std::size_t size)
      : Memory{ptr, size}
    {
    }

    not_pinned(not_pinned&&) noexcept = default;

    ~not_pinned() {}

protected:
    bool m_pinned = false;
};