#pragma once
#include <cstdint>
#include <vector>

#include "MathAssert.h"
#include "MathGeneral.h"
#include "BitmaskRank.h"

namespace JDK::Math
{
	/**
	 * @brief A single-level (flat) rank dictionary.
	 * Provides O(1) rank queries with maximum performance at the cost of a larger memory footprint.
	 * Requires sizeof(TRank) bytes of memory for every 64 bits of the original bitmask.
	 *
	 * @tparam TRank The unsigned integer type used to store cumulative ranks (default: uint32_t).
	*/
	template<typename TRank = uint32_t>
	class CFlatRankTable
	{
		static_assert(Internal::IsSupportedRankType<TRank>, "TRank must be uint8_t, uint16_t, uint32_t or uint64_t");

	public:
		/**
		 * @brief Builds the rank table based on the provided bitmask.
		 *
		 * @param bitmask The source bitmask array. The table stores a pointer to this bitmask,
		 *                so the bitmask array must outlive the rank table instance.
		*/
		void Build(const std::vector<uint64_t>& bitmask)
		{
			m_pBitmask = &bitmask;
			m_ranks.clear();
			m_totalRank = 0;

			if (bitmask.empty())
				return;

			m_ranks.resize(bitmask.size());
			BitmaskRank::Build<TRank>(*m_pBitmask, m_ranks, m_totalRank);
		}

		/**
		 * @brief Retrieves the absolute rank for a specific bit index.
		 *
		 * @param index The 0-based bit index to query.
		 * 
		 * @return The absolute rank of the bit. Returns std::numeric_limits<TRank>::max()
		 *         if the bit is unset (0) or the index is out of bounds.
		*/
		[[nodiscard]] inline uint64_t GetRank(uint64_t index) const noexcept
		{
			JDK_MATH_ASSERT(m_pBitmask != nullptr, "GetRank called before Build");

			const size_t blockIndex = index >> 6;
			if (blockIndex >= m_ranks.size())
				return std::numeric_limits<uint64_t>::max();

			const uint64_t bitOffset = index & 63;
			const uint64_t block = (*m_pBitmask)[blockIndex];

			if ((block & (1ULL << bitOffset)) == 0)
				return std::numeric_limits<uint64_t>::max();

			const uint64_t mask = (1ULL << bitOffset) - 1;
			return m_ranks[blockIndex] + static_cast<uint64_t>(BitMath::PopCount(block & mask));
		}

		/**
		 * @brief Resolves up to 8 bit indices into absolute ranks using a SIMD-optimized batch query.
		 *
		 * @param pIndices	Pointer to the input array of bit indices.
		 * @param pOutRanks	Pointer to the output array where ranks will be stored.
		 * @param count		The number of indices to process (must be strictly in the range [1, 8]).
		*/
		inline void GetRanksBatch(const uint64_t* pIndices, uint64_t* pOutRanks, size_t count) const noexcept
		{
			JDK_MATH_ASSERT(m_pBitmask != nullptr, "GetRanksBatch called before Build");
			BitmaskRank::GetRanksBatch<TRank>(*m_pBitmask, m_ranks.data(), pIndices, pOutRanks, count);
		}

		/**
		 * @brief Resolves up to 8 bit indices into absolute ranks using a SIMD-optimized batch query.
		 *
		 * @param indices	Input array of bit indices to query.
		 * @param outRanks	Output array where the computed ranks will be stored.
		*/
		template<size_t N_IN, size_t N_OUT>
		inline void GetRanksBatch(const uint64_t(&indices)[N_IN], uint64_t(&outRanks)[N_OUT]) const noexcept
		{
			static_assert(N_IN <= N_OUT, "Output array is too small");
			BitmaskRank::GetRanksBatch<TRank>(*m_pBitmask, m_ranks.data(), indices, outRanks, N_IN);
		}

		/**
		 * @brief Gets the total number of set bits (1s) across the entire bitmask.
		 *
		 * @return The total rank count.
		*/
		[[nodiscard]] inline uint64_t GetTotalRank() const noexcept
		{
			return static_cast<uint64_t>(m_totalRank);
		}

		/**
		 * @brief Calculates the total physical memory used by the internal cache tables.
		 *
		 * @return The memory usage in bytes.
		*/
		[[nodiscard]] inline size_t GetMemoryUsage() const noexcept
		{
			return m_ranks.capacity() * sizeof(TRank);
		}

	private:
		const std::vector<uint64_t>* m_pBitmask = nullptr;

		std::vector<TRank> m_ranks;
		TRank m_totalRank = 0;
	};
}