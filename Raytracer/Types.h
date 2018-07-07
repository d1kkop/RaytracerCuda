#pragma once

#include <memory>
#include <vector>
#include <map>


namespace Beam
{
    using i32 = __int32;
    using i64 = __int64;
    using u32 = unsigned __int32;
    using u64 = unsigned __int64;

    template <typename T> using sptr  = std::shared_ptr<T>;
    template <typename T> using wptr  = std::weak_ptr<T>;
    template <typename T> using Array = std::vector<T>;
    template <typename Key, typename Value> using Map = std::map<Key, Value>;
}
