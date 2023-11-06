template<class Memory>
/** @brief Host or device memory taht is not to be pinned. */
class not_pinned : public Memory {
public:
    not_pinned()
    : Memory{}
    {}
    
    not_pinned(Memory&& mem) 
    : Memory{std::move(mem)}
    {}

    not_pinned(const std::size_t size, std::size_t alignement = 0) 
    : Memory{size, alignement}
    {}

    not_pinned(not_pinned&&) noexcept = default;

    ~not_pinned() 
    {}

protected: 
    bool m_pinned = false;
};