#pragma once
#include <cstdint>
#include <variant>

#include "MathConfig.h"
#include "MathAssert.h"
#include "FlatRankTable.h"
#include "TwoLevelRankTable.h"

namespace JDK::Math
{
	/**
	 * @brief Universal rank table wrapper.
	 * Automatically selects the optimal table architecture (Flat vs. Two-Level)
	 * and the most memory-efficient data types based on the bitmask size.
	*/
	class CUniversalRankTable final
	{
	public:
		/**
		 * @brief Builds the optimal rank table based on the provided bitmask.
		 *
		 * @param bitmask The source bitmask array. The table stores a pointer to this bitmask.
		*/
		inline void Build(const std::vector<uint64_t>& bitmask)
		{
			if (bitmask.empty())
			{
				m_table.emplace<std::monostate>();
				return;
			}

			const uint64_t maxPossibleRank = static_cast<uint64_t>(bitmask.size()) * 64ULL;

			if (maxPossibleRank <= std::numeric_limits<uint8_t>::max())
			{
				m_table.emplace<TFlat8>().Build(bitmask);
			}
			else if (maxPossibleRank <= std::numeric_limits<uint16_t>::max())
			{
				if (bitmask.size() * sizeof(uint16_t) >= JDK_MATH_TWOLEVEL_RANK_THRESHOLD_BYTES)
					m_table.emplace<TTwo16>().Build(bitmask);
				else
					m_table.emplace<TFlat16>().Build(bitmask);
			}
			else if (maxPossibleRank <= std::numeric_limits<uint32_t>::max())
			{
				if (bitmask.size() * sizeof(uint32_t) >= JDK_MATH_TWOLEVEL_RANK_THRESHOLD_BYTES)
					m_table.emplace<TTwo32>().Build(bitmask);
				else
					m_table.emplace<TFlat32>().Build(bitmask);
			}
			else
			{
				if (bitmask.size() * sizeof(uint64_t) >= JDK_MATH_TWOLEVEL_RANK_THRESHOLD_BYTES)
					m_table.emplace<TTwo64>().Build(bitmask);
				else
					m_table.emplace<TFlat64>().Build(bitmask);
			}
		}

		/**
		 * @brief Retrieves the absolute rank for a specific bit index.
		 *
		 * @param index The 0-based bit index to query.
		 * 
		 * @return The absolute rank of the bit. Returns std::numeric_limits<uint64_t>::max()
		 *         if the bit is unset (0) or the index is out of bounds.
		*/
		inline uint64_t GetRank(uint64_t index) const noexcept
		{
			JDK_MATH_ASSERT(m_table.index() != 0, "GetRank called before Build");

			return std::visit([index](auto&& table) -> uint64_t
			{
				using T = std::decay_t<decltype(table)>;
				if constexpr (std::is_same_v<T, std::monostate>)
					return std::numeric_limits<uint64_t>::max();
				else
					return table.GetRank(index);
			}, m_table);
		}

		/**
		 * @brief Resolves up to 8 bit indices into absolute ranks.
		 *
		 * @param pIndices	Pointer to the input array of bit indices.
		 * @param pOutRanks	Pointer to the output array where ranks will be stored.
		 * @param count		The number of indices to process (must be strictly in the range [1, 8]).
		*/
		inline void GetRanksBatch(const uint64_t* pIndices, uint64_t* pOutRanks, size_t count) const noexcept
		{
			JDK_MATH_ASSERT(m_table.index() != 0, "GetRanksBatch called before Build");

			std::visit([&](auto&& table)
			{
				using T = std::decay_t<decltype(table)>;
				if constexpr (std::is_same_v<T, std::monostate>)
					for (size_t i = 0; i < count; ++i)
						pOutRanks[i] = std::numeric_limits<uint64_t>::max();
				else
					table.GetRanksBatch(pIndices, pOutRanks, count);
			}, m_table);
		}

		/**
		 * @brief Resolves up to 8 bit indices into absolute ranks.
		 *
		 * @param indices	Input array of bit indices to query.
		 * @param outRanks	Output array where the computed ranks will be stored.
		*/
		template<size_t N_IN, size_t N_OUT>
		inline void GetRanksBatch(const uint64_t(&indices)[N_IN], uint64_t(&outRanks)[N_OUT]) const noexcept
		{
			static_assert(N_IN <= N_OUT, "Output array is too small");
			GetRanksBatch(indices, outRanks, N_IN);
		}
		
		/**
		 * @brief Gets the total number of set bits (1s) across the entire bitmask.
		*/
		[[nodiscard]] inline uint64_t GetTotalRank() const noexcept
		{
			return std::visit([](auto&& table) -> uint64_t
			{
				using T = std::decay_t<decltype(table)>;
				if constexpr (std::is_same_v<T, std::monostate>)
					return 0;
				else
					return table.GetTotalRank();
			}, m_table);
		}

		/**
		 * @brief Calculates the total physical memory used by the internal cache table.
		*/
		[[nodiscard]] inline size_t GetMemoryUsage() const noexcept
		{
			return std::visit([](auto&& table) -> size_t
			{
				using T = std::decay_t<decltype(table)>;
				if constexpr (std::is_same_v<T, std::monostate>)
					return 0;
				else
					return table.GetMemoryUsage();
			}, m_table);
		}

	private:
		using TFlat8 = CFlatRankTable<uint8_t>;
		using TFlat16 = CFlatRankTable<uint16_t>;
		using TFlat32 = CFlatRankTable<uint32_t>;
		using TFlat64 = CFlatRankTable<uint64_t>;

		using TTwo16 = CTwoLevelRankTable<uint16_t, uint16_t>;
		using TTwo32 = CTwoLevelRankTable<uint32_t, uint16_t>;
		using TTwo64 = CTwoLevelRankTable<uint64_t, uint16_t>;

	private:
		std::variant<std::monostate, TFlat8, TFlat16, TFlat32, TFlat64, TTwo16, TTwo32, TTwo64> m_table;
	};
}