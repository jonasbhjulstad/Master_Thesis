#ifndef FIPOPT_MEMOIZER_HPP
#define FIPOPT_MEMOIZER_HPP
#include <iostream>
#include <Eigen/Dense>
#include <functional>
#include <Common/Circular_Buffer/Circular_Buffer.hpp>
namespace FIPOPT
{
    enum memoizer_status
    {
        DATA_NOT_MEMOIZED,
        DATA_MEMOIZED
    };

    template <typename Input, typename Return, int buffer_size>
    struct memoizer
    {
    private:
        jm::circular_buffer<std::pair<Input, Return>, buffer_size> buffer_;

        const std::string ID_;
        int buffer_iterator_ = 0;

    public:
        memoizer(const std::string &ID) : ID_(ID)
        {
        }

        inline bool Get_Data(const Input &x, Return &res)
        {
            for (auto &p : buffer_)
            {
                if (std::get<0>(p) == x)
                {
                    res = std::get<1>(p);
                    return DATA_MEMOIZED;
                }
            }
            return DATA_NOT_MEMOIZED;
        }

        void Set_Data(const Input &x, const Return &data)
        {

            buffer_.push_back(std::make_pair(x, data));

            buffer_iterator_ = buffer_iterator_ % buffer_size;
        }
    };
}

#endif