template <typename Simple_Resource>
class simple : public Simple_Resource
{
public:
    using resource_t = Simple_Resource;
    using this_type = simple<resource_t>;

    simple()
      : resource_t{}
    {
    }

    simple(const std::size_t size, const std::size_t alignment = 0)
      : resource_t{size, alignment}
    {
    }
};
