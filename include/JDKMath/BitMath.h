#pragma once
#include <cstdint>

#include "MathConfig.h"
#include "MathAssert.h"
#include "CpuFeatures.h"
#include "MathGeneral.h"

#if defined(_MSC_VER)
	#include <intrin.h>
#else
	#include <x86intrin.h>
#endif

namespace JDK::Math::BitMath
{
	namespace Internal
	{
		[[nodiscard]] inline uint32_t PopCount64_Software(uint64_t value) noexcept
		{
			value -= (value >> 1) & 0x5555555555555555ULL;
			value = (value & 0x3333333333333333ULL) + ((value >> 2) & 0x3333333333333333ULL);
			value = (value + (value >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
			return static_cast<uint32_t>((value * 0x0101010101010101ULL) >> 56);
		}

		[[nodiscard]] inline uint32_t PopCount32_Software(uint32_t value) noexcept
		{
			value -= (value >> 1) & 0x55555555;
			value = (value & 0x33333333) + ((value >> 2) & 0x33333333);
			value = (value + (value >> 4)) & 0x0F0F0F0F;
			return (value * 0x01010101) >> 24;
		}

		[[nodiscard]] inline uint32_t PopCount16_Hardware(uint16_t value) noexcept
		{
#if defined(_MSC_VER)
			return static_cast<uint32_t>(__popcnt16(value));
#else
			return static_cast<uint32_t>(__builtin_popcount(value));
#endif
		}

		[[nodiscard]] inline uint32_t PopCount64_Hardware(uint64_t value) noexcept
		{
#if defined(_MSC_VER)
			return static_cast<uint32_t>(__popcnt64(value));
#else
			return static_cast<uint32_t>(__builtin_popcountll(value));
#endif
		}

		[[nodiscard]] inline uint32_t PopCount32_Hardware(uint32_t value) noexcept
		{
#if defined(_MSC_VER)
			return static_cast<uint32_t>(__popcnt(value));
#else
			return static_cast<uint32_t>(__builtin_popcount(value));
#endif
		}
	}

	[[nodiscard]] inline uint32_t PopCount64(uint64_t value) noexcept
	{
#if JDK_MATH_DISPATCH_POPCNT == 1
		return Internal::PopCount64_Hardware(value);
#elif JDK_MATH_DISPATCH_POPCNT == 0
		return Internal::PopCount64_Software(value);
#else
		if (JDK::Math::GetCpuFeatures().bHasPopcnt)
			return Internal::PopCount64_Hardware(value);
		else
			return Internal::PopCount64_Software(value);
#endif
	}

	[[nodiscard]] inline uint32_t PopCount32(uint32_t value) noexcept
	{
#if JDK_MATH_DISPATCH_POPCNT == 1
		return Internal::PopCount32_Hardware(value);
#elif JDK_MATH_DISPATCH_POPCNT == 0
		return Internal::PopCount32_Software(value);
#else
		if (JDK::Math::GetCpuFeatures().bHasPopcnt)
			return Internal::PopCount32_Hardware(value);
		else
			return Internal::PopCount32_Software(value);
#endif
	}

	[[nodiscard]] inline uint32_t PopCount16(uint16_t value) noexcept
	{
#if JDK_MATH_DISPATCH_POPCNT == 1
		return Internal::PopCount16_Hardware(value);
#elif JDK_MATH_DISPATCH_POPCNT == 0
		return Internal::PopCount32_Software(value);
#else
		if (JDK::Math::GetCpuFeatures().bHasPopcnt)
			return Internal::PopCount16_Hardware(value);
		else
			return Internal::PopCount32_Software(value);
#endif
	}

	[[nodiscard]] inline uint32_t PopCount8(uint8_t value) noexcept
	{
		return PopCount16(value);
	}

	template<typename T>
	[[nodiscard]] inline uint32_t PopCount(T value) noexcept
	{
		static_assert(Math::Internal::IsSupportedRankType<T>, "TRank must be uint8_t, uint16_t, uint32_t or uint64_t");

		if constexpr (sizeof(T) == 8)
			return PopCount64(static_cast<uint64_t>(value));
		else if constexpr (sizeof(T) == 4)
			return PopCount32(static_cast<uint32_t>(value));
		else if constexpr (sizeof(T) == 2)
			return PopCount16(static_cast<uint16_t>(value));
		else if constexpr (sizeof(T) == 1)
			return PopCount8(static_cast<uint8_t>(value));
		else
			return 0;

	}
}