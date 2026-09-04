#pragma once
#include <cstdint>
#include <vector>

#include "MathAssert.h"
#include "MathGeneral.h"
#include "BitmaskRank.h"

namespace JDK::Math
{
	/**
	 * @brief A two-level (L1/L2) rank dictionary.
	 * Reduces memory usage by caching absolute ranks (SuperRanks) every 32 blocks (2048 bits)
	 * and relative offsets (LocalRanks) for each individual 64-bit block.
	 *
	 * @tparam TSuperRank The unsigned integer type for absolute L1 sums (default: uint32_t).
	 * @tparam TLocalRank The unsigned integer type for relative L2 offsets (default: uint16_t).
	*/
	template<typename TSuperRank = uint32_t, typename TLocalRank = uint16_t>
	class CTwoLevelRankTable
	{
		static_assert(Internal::IsSupportedRankType<TSuperRank>, "TSuperRank must be uint8_t, uint16_t, uint32_t or uint64_t");
		static_assert(Internal::IsSupportedRankType<TLocalRank>, "TLocalRank must be uint8_t, uint16_t, uint32_t or uint64_t");

	public:
		/**
		 * @brief Builds the two-level rank table based on the provided bitmask.
		 *
		 * @param bitmask The source bitmask array. The table stores a pointer to this bitmask,
		 *                so the bitmask array must outlive the rank table instance.
		*/
		void Build(const std::vector<uint64_t>& bitmask)
		{
			m_pBitmask = &bitmask;
			m_superRanks.clear();
			m_localRanks.clear();
			m_totalRank = 0;

			if (bitmask.empty())
				return;

			const size_t count = bitmask.size();
			m_superRanks.resize((count + 31) / 32);
			m_localRanks.resize(count);

			BitmaskRank::BuildTwoLevel<TSuperRank, TLocalRank>(*m_pBitmask, m_superRanks, m_localRanks, m_totalRank);
		}

		/**
		 * @brief Retrieves the absolute rank for a specific bit index.
		 *
		 * @param index The 0-based bit index to query.
		 * 
		 * @return The absolute rank of the bit. Returns std::numeric_limits<TSuperRank>::max()
		 *         if the bit is unset (0) or the index is out of bounds.
		*/
		[[nodiscard]] inline uint64_t GetRank(uint64_t index) const noexcept
		{
			JDK_MATH_ASSERT(m_pBitmask != nullptr, "GetRank called before Build");

			const size_t blockIndex = index >> 6;
			if (blockIndex >= m_localRanks.size())
				return std::numeric_limits<uint64_t>::max();

			const uint64_t bitOffset = index & 63;
			const uint64_t block = (*m_pBitmask)[blockIndex];

			if ((block & (1ULL << bitOffset)) == 0)
				return std::numeric_limits<uint64_t>::max();

			const size_t superIndex = blockIndex >> 5; // 32 blocks per super rank
			const uint64_t mask = (1ULL << bitOffset) - 1;

			return m_superRanks[superIndex]
				+ static_cast<uint64_t>(m_localRanks[blockIndex])
				+ static_cast<uint64_t>(BitMath::PopCount(block & mask));
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
			BitmaskRank::GetTwoLevelRanksBatch<TSuperRank, TLocalRank>(*m_pBitmask, m_superRanks.data(), m_localRanks.data(), pIndices, pOutRanks, count);
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
			BitmaskRank::GetTwoLevelRanksBatch<TSuperRank, TLocalRank>(*m_pBitmask, m_superRanks.data(), m_localRanks.data(), indices, outRanks, N_IN);
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
		 * @brief Calculates the total physical memory used by both L1 and L2 cache tables.
		 *
		 * @return The memory usage in bytes.
		*/
		[[nodiscard]] inline size_t GetMemoryUsage() const noexcept
		{
			return (m_superRanks.capacity() * sizeof(TSuperRank)) + (m_localRanks.capacity() * sizeof(TLocalRank));
		}

	private:
		const std::vector<uint64_t>* m_pBitmask = nullptr;

		std::vector<TSuperRank> m_superRanks;
		std::vector<TLocalRank> m_localRanks;
		TSuperRank m_totalRank = 0;
	};
}