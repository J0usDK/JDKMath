#pragma once
#include <cstdint>
#include <type_traits>

namespace JDK::Math
{
	namespace Internal
	{
		template<typename T>
		inline constexpr bool IsSupportedRankType =
			std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>;
	}
}