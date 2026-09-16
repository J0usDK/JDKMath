#pragma once
#include <cstdint>
#include <vector>
#include <iterator>
#include <limits>

#include "MathConfig.h"
#include "MathAssert.h"
#include "CpuFeatures.h"
#include "MathGeneral.h"
#include "BitMath.h"

#if defined(_MSC_VER)
	#include <intrin.h>
	#include <immintrin.h>
#else
	#include <x86intrin.h>
#endif

namespace JDK::Math::BitmaskRank
{
	namespace Internal
	{
		template<typename TRank>
		inline void BuildRanks_Scalar(const uint64_t* pBitmask, size_t count, TRank* pRanks, TRank& outTotalRank) noexcept
		{
			TRank currentRank = 0;
			for (size_t i = 0; i < count; ++i)
			{
				pRanks[i] = currentRank;
				currentRank += static_cast<TRank>(BitMath::PopCount(pBitmask[i]));
			}
			outTotalRank = currentRank;
		}

		template<typename TRank>
#if defined(__GNUC__) || defined(__clang__)
		__attribute__((target("avx512f,avx512vpopcntdq")))
#endif
		inline void BuildRanks_AVX512(const uint64_t* pBitmask, size_t count, TRank* pRanks, TRank& outTotalRank) noexcept
		{
			if constexpr (sizeof(TRank) > 8)
			{
				BuildRanks_Scalar<TRank>(pBitmask, count, pRanks, outTotalRank);
				return;
			}
			else
			{
				TRank globalRank = 0;
				size_t i = 0;

				const __m512i perm1 = _mm512_setr_epi64(0, 0, 1, 2, 3, 4, 5, 6);
				const __m512i perm2 = _mm512_setr_epi64(0, 0, 0, 1, 2, 3, 4, 5);
				const __m512i perm4 = _mm512_setr_epi64(0, 0, 0, 0, 0, 1, 2, 3);

				const size_t vectorizedCount = count & ~size_t(7);

				for (; i < vectorizedCount; i += 8)
				{
					__m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(pBitmask + i));
					__m512i popcnts = _mm512_popcnt_epi64(v);

					__m512i s1 = _mm512_maskz_permutexvar_epi64(0xFE, perm1, popcnts);
					__m512i x1 = _mm512_add_epi64(popcnts, s1);

					__m512i s2 = _mm512_maskz_permutexvar_epi64(0xFC, perm2, x1);
					__m512i x2 = _mm512_add_epi64(x1, s2);

					__m512i s4 = _mm512_maskz_permutexvar_epi64(0xF0, perm4, x2);

					__m512i inclusiveSum = _mm512_add_epi64(x2, s4);
					__m512i exclusiveSum = _mm512_maskz_permutexvar_epi64(0xFE, perm1, inclusiveSum);

					__m512i globalBaseVec = _mm512_set1_epi64(static_cast<uint64_t>(globalRank));
					__m512i finalRanks64 = _mm512_add_epi64(exclusiveSum, globalBaseVec);

					if constexpr (sizeof(TRank) == 8)
					{
						_mm512_storeu_si512(reinterpret_cast<__m512i*>(pRanks + i), finalRanks64);
					}
					else if constexpr (sizeof(TRank) == 4)
					{
						__m256i ranks32 = _mm512_cvtepi64_epi32(finalRanks64);
						_mm256_storeu_si256(reinterpret_cast<__m256i*>(pRanks + i), ranks32);
					}
					else if constexpr (sizeof(TRank) == 2)
					{
						__m128i ranks16 = _mm512_cvtepi64_epi16(finalRanks64);
						_mm_storeu_si128(reinterpret_cast<__m128i*>(pRanks + i), ranks16);
					}
					else if constexpr (sizeof(TRank) == 1)
					{
						__m128i ranks8 = _mm512_cvtepi64_epi8(finalRanks64);
						_mm_storel_epi64(reinterpret_cast<__m128i*>(pRanks + i), ranks8);
					}

					globalRank += static_cast<TRank>(_mm512_reduce_add_epi64(popcnts));
				}

				for (; i < count; ++i)
				{
					pRanks[i] = globalRank;
					globalRank += static_cast<TRank>(BitMath::PopCount(pBitmask[i]));
				}

				outTotalRank = globalRank;
			}
		}

		template<typename TSuperRank, typename TLocalRank>
		inline void BuildTwoLevelRanks_Scalar(const uint64_t* pBitmask, size_t count, TSuperRank* pSuperRanks, TLocalRank* pLocalRanks, TSuperRank& outTotalRank) noexcept
		{
			TSuperRank globalRank = 0;
			TSuperRank localRank = 0;

			for (size_t i = 0; i < count; ++i)
			{
				if ((i & 31) == 0)
				{
					pSuperRanks[i >> 5] = globalRank;
					localRank = 0;
				}

				pLocalRanks[i] = static_cast<TLocalRank>(localRank);

				uint32_t popCount = BitMath::PopCount(pBitmask[i]);
				localRank += static_cast<TLocalRank>(popCount);
				globalRank += static_cast<TSuperRank>(popCount);
			}
			outTotalRank = globalRank;
		}

		template<typename TSuperRank, typename TLocalRank>
#if defined(__GNUC__) || defined(__clang__)
		__attribute__((target("avx512f,avx512vpopcntdq")))
#endif
		inline void BuildTwoLevelRanks_AVX512(const uint64_t* pBitmask, size_t count, TSuperRank* pSuperRanks, TLocalRank* pLocalRanks, TSuperRank& outTotalRank) noexcept
		{
			TSuperRank globalRank = 0;
			size_t i = 0;

			const size_t superBlockSize = 32;
			const size_t vectorizedCount = count & ~size_t(31);

			const __m512i perm1 = _mm512_setr_epi64(0, 0, 1, 2, 3, 4, 5, 6);
			const __m512i perm2 = _mm512_setr_epi64(0, 0, 0, 1, 2, 3, 4, 5);
			const __m512i perm4 = _mm512_setr_epi64(0, 0, 0, 0, 0, 1, 2, 3);

			for (; i < vectorizedCount; i += superBlockSize)
			{
				pSuperRanks[i >> 5] = globalRank;
				TSuperRank currentLocalRank = 0;

				for (size_t step = 0; step < 4; ++step)
				{
					size_t offset = i + step * 8;

					__m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(pBitmask + offset));
					__m512i popcnts = _mm512_popcnt_epi64(v);

					__m512i s1 = _mm512_maskz_permutexvar_epi64(0xFE, perm1, popcnts);
					__m512i x1 = _mm512_add_epi64(popcnts, s1);

					__m512i s2 = _mm512_maskz_permutexvar_epi64(0xFC, perm2, x1);
					__m512i x2 = _mm512_add_epi64(x1, s2);

					__m512i s4 = _mm512_maskz_permutexvar_epi64(0xF0, perm4, x2);
					__m512i inclusiveSum = _mm512_add_epi64(x2, s4);
					__m512i exclusiveSum = _mm512_maskz_permutexvar_epi64(0xFE, perm1, inclusiveSum);

					__m512i baseVec = _mm512_set1_epi64(static_cast<uint64_t>(currentLocalRank));
					__m512i finalRanks64 = _mm512_add_epi64(exclusiveSum, baseVec);

					if constexpr (sizeof(TLocalRank) == 8)
						_mm512_storeu_si512(reinterpret_cast<__m512i*>(pLocalRanks + offset), finalRanks64);
					else if constexpr (sizeof(TLocalRank) == 4)
						_mm256_storeu_si256(reinterpret_cast<__m256i*>(pLocalRanks + offset), _mm512_cvtepi64_epi32(finalRanks64));
					else if constexpr (sizeof(TLocalRank) == 2)
						_mm_storeu_si128(reinterpret_cast<__m128i*>(pLocalRanks + offset), _mm512_cvtepi64_epi16(finalRanks64));
					else if constexpr (sizeof(TLocalRank) == 1)
					{
						__m128i ranks8 = _mm512_cvtepi64_epi8(finalRanks64);
						_mm_storel_epi64(reinterpret_cast<__m128i*>(pLocalRanks + offset), ranks8);
					}

					currentLocalRank += static_cast<TSuperRank>(_mm512_reduce_add_epi64(popcnts));
				}
				globalRank += currentLocalRank;
			}

			TSuperRank localTailRank = 0;
			for (; i < count; ++i)
			{
				if ((i & 31) == 0)
				{
					pSuperRanks[i >> 5] = globalRank;
					localTailRank = 0;
				}

				pLocalRanks[i] = static_cast<TLocalRank>(localTailRank);

				uint32_t popCount = BitMath::PopCount(pBitmask[i]);
				localTailRank += popCount;
				globalRank += popCount;
			}
			outTotalRank = globalRank;
		}

		template<typename TRank>
		inline void GetRanksBatch_Scalar(const uint64_t* pBitmask, const TRank* pRanks, const uint64_t* pIndices, uint64_t* pOutRanks, size_t count) noexcept
		{
			for (size_t i = 0; i < count; ++i)
			{
				const uint64_t index = pIndices[i];
				const size_t blockIndex = index >> 6;
				const uint64_t bitOffset = index & 63;
				const uint64_t block = pBitmask[blockIndex];

				if ((block & (1ULL << bitOffset)) == 0)
					pOutRanks[i] = std::numeric_limits<uint64_t>::max();
				else
				{
					const uint64_t mask = (1ULL << bitOffset) - 1;
					pOutRanks[i] = pRanks[blockIndex] + static_cast<uint64_t>(BitMath::PopCount64(block & mask));
				}
			}
		}

		template<typename TRank>
#if defined(__GNUC__) || defined(__clang__)
		__attribute__((target("avx512f,avx512vl,avx512bw,avx512vpopcntdq")))
#endif
		inline void GetRanksBatch_AVX512(const uint64_t* pBitmask, const TRank* pRanks, size_t safeLimit, const uint64_t* pIndices, uint64_t* pOutRanks, size_t count) noexcept
		{
			const __mmask8 loadMask = (1 << count) - 1;

			__m512i indices = _mm512_maskz_loadu_epi64(loadMask, pIndices);

			__m512i blockIndices = _mm512_srli_epi64(indices, 6);
			__m512i bitOffsets = _mm512_and_si512(indices, _mm512_set1_epi64(63));

			__m512i blocks = _mm512_mask_i64gather_epi64(_mm512_setzero_si512(), loadMask, blockIndices, pBitmask, 8);

			__m512i one = _mm512_set1_epi64(1);
			__m512i bitFlags = _mm512_sllv_epi64(one, bitOffsets);
			__m512i existMaskVec = _mm512_and_si512(blocks, bitFlags);

			__mmask8 validMask = _mm512_test_epi64_mask(existMaskVec, existMaskVec) & loadMask;

			__m512i bitMasks = _mm512_sub_epi64(bitFlags, one);
			__m512i maskedBlocks = _mm512_and_si512(blocks, bitMasks);
			__m512i localRanks64 = _mm512_popcnt_epi64(maskedBlocks);

			__m512i baseRanks64;

			if constexpr (sizeof(TRank) == 8)
			{
				baseRanks64 = _mm512_mask_i64gather_epi64(_mm512_setzero_si512(), validMask, blockIndices, pRanks, 8);
			}
			else
			{
				__mmask8 safeMask = validMask;
				__mmask8 unsafeMask = 0;

				if constexpr (sizeof(TRank) < 4)
				{
					__m512i vecSafeLimit = _mm512_set1_epi64(safeLimit);
					unsafeMask = _mm512_cmpge_epi64_mask(blockIndices, vecSafeLimit) & validMask;
					safeMask = validMask & ~unsafeMask;
				}

				__m256i baseRanks32 = _mm512_mask_i64gather_epi32(_mm256_setzero_si256(), safeMask, blockIndices, pRanks, sizeof(TRank));

				if constexpr (sizeof(TRank) < 4)
				{
					alignas(64) uint64_t bIdxArr[8];
					alignas(32) uint32_t rArr[8];
					_mm512_store_si512(bIdxArr, blockIndices);
					_mm256_store_si256(reinterpret_cast<__m256i*>(rArr), baseRanks32);

					for (size_t k = 0; k < count; ++k)
						if ((unsafeMask >> k) & 1)
							rArr[k] = static_cast<uint32_t>(pRanks[bIdxArr[k]]);

					baseRanks32 = _mm256_load_si256(reinterpret_cast<const __m256i*>(rArr));

					if constexpr (sizeof(TRank) == 2)
						baseRanks32 = _mm256_and_si256(baseRanks32, _mm256_set1_epi32(0xFFFF));
					else if constexpr (sizeof(TRank) == 1)
						baseRanks32 = _mm256_and_si256(baseRanks32, _mm256_set1_epi32(0xFF));
				}

				baseRanks64 = _mm512_cvtepu32_epi64(baseRanks32);
			}

			__m512i finalRanks64 = _mm512_add_epi64(baseRanks64, localRanks64);
			const TRank invalidValue = std::numeric_limits<uint64_t>::max();
			__m512i invalidFill = _mm512_set1_epi64(invalidValue);
			__m512i result = _mm512_mask_blend_epi64(validMask, invalidFill, finalRanks64);
			_mm512_mask_storeu_epi64(pOutRanks, loadMask, result);
		}

		template<typename TSuperRank, typename TLocalRank>
		inline void GetTwoLevelRanksBatch_Scalar(const uint64_t* pBitmask, const TSuperRank* pSuperRanks, const TLocalRank* pLocalRanks, const uint64_t* pIndices, uint64_t* pOutRanks, size_t count) noexcept
		{
			for (size_t i = 0; i < count; ++i)
			{
				const uint64_t index = pIndices[i];
				const size_t blockIndex = index >> 6;
				const uint64_t bitOffset = index & 63;
				const uint64_t block = pBitmask[blockIndex];

				if ((block & (1ULL << bitOffset)) == 0)
					pOutRanks[i] = std::numeric_limits<uint64_t>::max();
				else
				{
					const size_t superIndex = blockIndex >> 5;
					const uint64_t mask = (1ULL << bitOffset) - 1;

					pOutRanks[i] = pSuperRanks[superIndex]
						+ static_cast<uint64_t>(pLocalRanks[blockIndex])
						+ static_cast<uint64_t>(BitMath::PopCount(block & mask));
				}
			}
		}

		template<typename TSuperRank, typename TLocalRank>
#if defined(__GNUC__) || defined(__clang__)
		__attribute__((target("avx512f,avx512vl,avx512bw,avx512vpopcntdq")))
#endif
		inline void GetTwoLevelRanksBatch_AVX512(const uint64_t* pBitmask, const TSuperRank* pSuperRanks, size_t superRanksSafeLimit, const TLocalRank* pLocalRanks, size_t localRanksSafeLimit, const uint64_t* pIndices, uint64_t* pOutRanks, size_t count) noexcept
		{
			const __mmask8 loadMask = (1 << count) - 1;

			__m512i indices = _mm512_maskz_loadu_epi64(loadMask, pIndices);
			__m512i blockIndices = _mm512_srli_epi64(indices, 6);
			__m512i superIndices = _mm512_srli_epi64(blockIndices, 5);
			__m512i bitOffsets = _mm512_and_si512(indices, _mm512_set1_epi64(63));

			__m512i blocks = _mm512_mask_i64gather_epi64(_mm512_setzero_si512(), loadMask, blockIndices, pBitmask, 8);

			__m512i one = _mm512_set1_epi64(1);
			__m512i bitFlags = _mm512_sllv_epi64(one, bitOffsets);
			__m512i existMaskVec = _mm512_and_si512(blocks, bitFlags);
			__mmask8 validMask = _mm512_test_epi64_mask(existMaskVec, existMaskVec) & loadMask;

			__m512i bitMasks = _mm512_sub_epi64(bitFlags, one);
			__m512i maskedBlocks = _mm512_and_si512(blocks, bitMasks);
			__m512i popcntRanks64 = _mm512_popcnt_epi64(maskedBlocks);

			__m512i superRanks64;
			if constexpr (sizeof(TSuperRank) == 8)
				superRanks64 = _mm512_mask_i64gather_epi64(_mm512_setzero_si512(), validMask, superIndices, pSuperRanks, 8);
			else
			{
				__mmask8 safeSuperMask = validMask;
				__mmask8 unsafeSuperMask = 0;

				if constexpr (sizeof(TSuperRank) < 4)
				{
					__m512i vecSafeLimit = _mm512_set1_epi64(superRanksSafeLimit >> 5);
					unsafeSuperMask = _mm512_cmpge_epi64_mask(superIndices, vecSafeLimit) & validMask;
					safeSuperMask = validMask & ~unsafeSuperMask;
				}

				__m256i sr32 = _mm512_mask_i64gather_epi32(_mm256_setzero_si256(), safeSuperMask, superIndices, pSuperRanks, sizeof(TSuperRank));
				
				if constexpr (sizeof(TSuperRank) < 4)
				{
					if (unsafeSuperMask != 0)
					{
						alignas(64) uint64_t sIdxArr[8];
						alignas(32) uint32_t srArr[8];
						_mm512_store_si512(sIdxArr, superIndices);
						_mm256_store_si256(reinterpret_cast<__m256i*>(srArr), sr32);

						for (size_t k = 0; k < count; ++k)
							if ((unsafeSuperMask >> k) & 1)
								srArr[k] = static_cast<uint32_t>(pSuperRanks[sIdxArr[k]]);

						sr32 = _mm256_load_si256(reinterpret_cast<const __m256i*>(srArr));
					}

					if constexpr (sizeof(TSuperRank) == 2)
						sr32 = _mm256_and_si256(sr32, _mm256_set1_epi32(0xFFFF));
					if constexpr (sizeof(TSuperRank) == 1)
						sr32 = _mm256_and_si256(sr32, _mm256_set1_epi32(0xFF));
				}
				superRanks64 = _mm512_cvtepu32_epi64(sr32);
			}

			__m512i localRanks64;
			if constexpr (sizeof(TLocalRank) == 8)
				localRanks64 = _mm512_mask_i64gather_epi64(_mm512_setzero_si512(), validMask, blockIndices, pLocalRanks, 8);
			else
			{
				__mmask8 safeLocalMask = validMask;
				__mmask8 unsafeLocalMask = 0;

				if constexpr (sizeof(TLocalRank) < 4)
				{
					__m512i vecSafeLimit = _mm512_set1_epi64(localRanksSafeLimit);
					unsafeLocalMask = _mm512_cmpge_epi64_mask(blockIndices, vecSafeLimit) & validMask;
					safeLocalMask = validMask & ~unsafeLocalMask;
				}

				__m256i lr32 = _mm512_mask_i64gather_epi32(_mm256_setzero_si256(), safeLocalMask, blockIndices, pLocalRanks, sizeof(TLocalRank));
				
				if constexpr (sizeof(TLocalRank) < 4)
				{
					if (unsafeLocalMask != 0)
					{
						alignas(64) uint64_t bIdxArr[8];
						alignas(32) uint32_t lrArr[8];
						_mm512_store_si512(bIdxArr, blockIndices);
						_mm256_store_si256(reinterpret_cast<__m256i*>(lrArr), lr32);

						for (size_t k = 0; k < count; ++k)
							if ((unsafeLocalMask >> k) & 1)
								lrArr[k] = static_cast<uint32_t>(pLocalRanks[bIdxArr[k]]);

						lr32 = _mm256_load_si256(reinterpret_cast<const __m256i*>(lrArr));
					}

					if constexpr (sizeof(TLocalRank) == 2)
						lr32 = _mm256_and_si256(lr32, _mm256_set1_epi32(0xFFFF));
					if constexpr (sizeof(TLocalRank) == 1)
						lr32 = _mm256_and_si256(lr32, _mm256_set1_epi32(0xFF));
				}

				localRanks64 = _mm512_cvtepu32_epi64(lr32);
			}

			__m512i finalRanks64 = _mm512_add_epi64(superRanks64, localRanks64);
			finalRanks64 = _mm512_add_epi64(finalRanks64, popcntRanks64);

			const TSuperRank invalidValue = std::numeric_limits<uint64_t>::max();
			__m512i invalidFill = _mm512_set1_epi64(invalidValue);
			__m512i result = _mm512_mask_blend_epi64(validMask, invalidFill, finalRanks64);
			_mm512_mask_storeu_epi64(pOutRanks, loadMask, result);
		}
	}

	// ========================================================================
	// PUBLIC API
	// ========================================================================

	[[nodiscard]] inline bool CanUseBuildAVX512() noexcept
	{
		const auto& cpuFeatures = GetCpuFeatures();
		return cpuFeatures.bHasAVX512F && cpuFeatures.bHasAVX512_VPOPCNTDQ;
	}

	[[nodiscard]] inline bool CanUseGetBatchAVX512() noexcept
	{
		const auto& cpuFeatures = GetCpuFeatures();
		return cpuFeatures.bHasAVX512F && cpuFeatures.bHasAVX512VL && cpuFeatures.bHasAVX512BW && cpuFeatures.bHasAVX512_VPOPCNTDQ;
	}

	/**
	 * @brief Builds a cumulative rank table from a given bitmask array.
	 * 
	 * @tparam TRank Unsigned rank type. Must be uint8_t, uint16_t, uint32_t, or uint64_t.
	 *
	 * @param bitmask			Input array bitmask divided into 64-bit blocks.
	 * @param ranks				Output rank table. Must contain at least bitmask.size() elements.
	 * @param outTotalRank		Output parameter that will store the total number of set bits.
	 * 
	 * @note TRank must be able to represent the maximum possible rank.
	*/
	template<typename TRank = uint32_t>
	inline void Build(const std::vector<uint64_t>& bitmask, std::vector<TRank>& ranks, TRank& outTotalRank) noexcept
	{
		static_assert(Math::Internal::IsSupportedRankType<TRank>, "TRank must be uint8_t, uint16_t, uint32_t or uint64_t");

		if (bitmask.empty())
		{
			outTotalRank = 0;
			return;
		}

		JDK_MATH_ASSERT(ranks.size() >= bitmask.size(), "JDKMath: 'ranks' vector must be at least as large as 'bitmask' vector");
		if (ranks.size() < bitmask.size())
		{
			outTotalRank = 0;
			return;
		}

#if JDK_MATH_DISPATCH_AVX512_RANK_BUILD == 1
		Internal::BuildRanks_AVX512<TRank>(bitmask.data(), bitmask.size(), ranks.data(), outTotalRank);
#elif JDK_MATH_DISPATCH_AVX512_RANK_BUILD == 0
		Internal::BuildRanks_Scalar<TRank>(bitmask.data(), bitmask.size(), ranks.data(), outTotalRank);
#else
		if (CanUseBuildAVX512())
			Internal::BuildRanks_AVX512<TRank>(bitmask.data(), bitmask.size(), ranks.data(), outTotalRank);
		else
			Internal::BuildRanks_Scalar<TRank>(bitmask.data(), bitmask.size(), ranks.data(), outTotalRank);
#endif
	}

	/**
	 * @brief Builds a two-level cumulative rank table from a given bitmask array.
	 * 
	 * @tparam TSuperRank Unsigned rank type for super ranks. Must be uint8_t, uint16_t, uint32_t, or uint64_t.
	 * @tparam TLocalRank Unsigned rank type for local ranks. Must be uint8_t, uint16_t, uint32_t, or uint64_t.
	 *
	 * @param bitmask			Input array bitmask divided into 64-bit blocks.
	 * @param superRanks		Output super rank table. Must contain at least (bitmask.size() + 31) / 32 elements.
	 * @param localRanks		Output local rank table. Must contain at least bitmask.size() elements.
	 * @param outTotalRank		Output parameter that will store the total number of set bits.
	 *
	 * @note TSuperRank and TLocalRank must be able to represent the maximum possible rank.
	*/
	template<typename TSuperRank = uint32_t, typename TLocalRank = uint16_t>
	inline void BuildTwoLevel(const std::vector<uint64_t>& bitmask, std::vector<TSuperRank>& superRanks, std::vector<TLocalRank>& localRanks, TSuperRank& outTotalRank) noexcept
	{
		static_assert(Math::Internal::IsSupportedRankType<TSuperRank>, "TSuperRank must be uint8_t, uint16_t, uint32_t or uint64_t");
		static_assert(Math::Internal::IsSupportedRankType<TLocalRank>, "TLocalRank must be uint8_t, uint16_t, uint32_t or uint64_t");

		if (bitmask.empty())
		{
			outTotalRank = 0;
			return;
		}

		JDK_MATH_ASSERT(localRanks.size() >= bitmask.size(), "JDKMath: 'localRanks' vector must be at least as large as 'bitmask' vector");
		if (localRanks.size() < bitmask.size())
		{
			outTotalRank = 0;
			return;
		}

#if JDK_MATH_DISPATCH_AVX512_RANK_BUILD == 1
		Internal::BuildTwoLevelRanks_AVX512<TSuperRank, TLocalRank>(bitmask.data(), bitmask.size(), superRanks.data(), localRanks.data(), outTotalRank);
#elif JDK_MATH_DISPATCH_AVX512_RANK_BUILD == 0
		Internal::BuildTwoLevelRanks_Scalar<TSuperRank, TLocalRank>(bitmask.data(), bitmask.size(), superRanks.data(), localRanks.data(), outTotalRank);
#else
		if (CanUseBuildAVX512())
			Internal::BuildTwoLevelRanks_AVX512<TSuperRank, TLocalRank>(bitmask.data(), bitmask.size(), superRanks.data(), localRanks.data(), outTotalRank);
		else
			Internal::BuildTwoLevelRanks_Scalar<TSuperRank, TLocalRank>(bitmask.data(), bitmask.size(), superRanks.data(), localRanks.data(), outTotalRank);
#endif
	}

	/**
	 * @brief Resolves up to 8 bit indices into absolute ranks.
	 *
	 * @tparam TRank Unsigned rank type. Must be uint8_t, uint16_t, uint32_t, or uint64_t.
	 * 
	 * @param bitmask		Source bitmask.
	 * @param pRanks		The cumulative rank table computed by Build().
	 * @param ranksCapacity	The physical memory (in elements) of the pRanks array.
	 * @param pIndices		Input array of bit indices to query.
	 * @param pOutRanks		Output array where the computed ranks will be stored.
	 *						Missing or out-of-bounds indices will yield std::numeric_limits<uint64_t>::max().
	 * @param count			The number of indices to process. Must be in [1, 8].
	*/
	template<typename TRank = uint32_t>
	inline void GetRanksBatch(const std::vector<uint64_t>& bitmask, const TRank* pRanks, size_t ranksCapacity, const uint64_t* pIndices, uint64_t* pOutRanks, size_t count) noexcept
	{
		static_assert(Math::Internal::IsSupportedRankType<TRank>, "TRank must be uint8_t, uint16_t, uint32_t or uint64_t");
		JDK_MATH_ASSERT(count > 0 && count <= 8, "JDKMath: GetRanksBatch supports strict count from 1 to 8 elements");
		JDK_MATH_ASSERT(!bitmask.empty(), "JDKMath: GetRanksBatch must receive not empty bitmask");

		size_t safeLimit = ranksCapacity - 1;

		const uintptr_t lastElementAddr = reinterpret_cast<uintptr_t>(pRanks + safeLimit);
		safeLimit += ((lastElementAddr & 4095) <= 4092);

#if JDK_MATH_DISPATCH_AVX512_RANK_GETBATCH == 1
		Internal::GetRanksBatch_AVX512<TRank>(bitmask.data(), pRanks, safeLimit, pIndices, pOutRanks, count);
#elif JDK_MATH_DISPATCH_AVX512_RANK_GETBATCH == 0
		Internal::GetRanksBatch_Scalar<TRank>(bitmask.data(), pRanks, pIndices, pOutRanks, count);
#else
		if (CanUseGetBatchAVX512())
			Internal::GetRanksBatch_AVX512<TRank>(bitmask.data(), pRanks, safeLimit, pIndices, pOutRanks, count);
		else
			Internal::GetRanksBatch_Scalar<TRank>(bitmask.data(), pRanks, pIndices, pOutRanks, count);
#endif
	}

	/**
	 * @brief Resolves up to 8 bit indices into absolute ranks using a two-level table.
	 *
	 * @tparam TSuperRank Unsigned rank type. Must be uint8_t, uint16_t, uint32_t, or uint64_t.
	 * @tparam TLocalRank Unsigned rank type. Must be uint8_t, uint16_t, uint32_t, or uint64_t.
	 *
	 * @param bitmask				bitmask Source bitmask.
	 * @param pSuperRanks			The super rank table computed by BuildTwoLevel().
	 * @param superRanksCapacity	The physical memory capacity (in elements) of the pSuperRanks array.
	 * @param pLocalRanks			The local rank table computed by BuildTwoLevel().
	 * @param localRanksCapacity	The physical memory capacity (in elements) of the pLocalRanks array.
	 * @param pIndices				Input array of bit indices to query.
	 * @param pOutRanks				Output array where the computed ranks will be stored.
	 *								Missing or out-of-bounds indices will yield std::numeric_limits<uint64_t>::max().
	 * @param count					The number of indices to process. Must be in [1, 8].
	*/
	template<typename TSuperRank = uint32_t, typename TLocalRank = uint16_t>
	inline void GetTwoLevelRanksBatch(const std::vector<uint64_t>& bitmask, const TSuperRank* pSuperRanks, size_t superRanksCapacity, const TLocalRank* pLocalRanks, size_t localRanksCapacity, const uint64_t* pIndices, uint64_t* pOutRanks, size_t count) noexcept
	{
		static_assert(Math::Internal::IsSupportedRankType<TSuperRank>, "TSuperRank must be uint8_t, uint16_t, uint32_t or uint64_t");
		static_assert(Math::Internal::IsSupportedRankType<TLocalRank>, "TLocalRank must be uint8_t, uint16_t, uint32_t or uint64_t");
		JDK_MATH_ASSERT(count > 0 && count <= 8, "JDKMath: GetTwoLevelRanksBatch supports strict count from 1 to 8 elements");
		JDK_MATH_ASSERT(!bitmask.empty(), "JDKMath: GetTwoLevelRanksBatch must receive not empty bitmask");

		size_t safeLocalLimit = localRanksCapacity - 1;
		size_t safeSuperLimit = superRanksCapacity - 1;

		const uintptr_t lastLocalAddr = reinterpret_cast<uintptr_t>(pLocalRanks + safeLocalLimit);
		safeLocalLimit += ((lastLocalAddr & 4095) <= 4092);

		const uintptr_t lastSuperAddr = reinterpret_cast<uintptr_t>(pSuperRanks + safeSuperLimit);
		safeSuperLimit += ((lastSuperAddr & 4095) <= 4092);

#if JDK_MATH_DISPATCH_AVX512_RANK_GETBATCH == 1
		Internal::GetTwoLevelRanksBatch_AVX512<TSuperRank, TLocalRank>(bitmask.data(), pSuperRanks, safeSuperLimit, pLocalRanks, safeLocalLimit, pIndices, pOutRanks, count);
#elif JDK_MATH_DISPATCH_AVX512_RANK_GETBATCH == 0
		Internal::GetTwoLevelRanksBatch_Scalar<TSuperRank, TLocalRank>(bitmask.data(), pSuperRanks, pLocalRanks, pIndices, pOutRanks, count);
#else
		if (CanUseGetBatchAVX512())
			Internal::GetTwoLevelRanksBatch_AVX512<TSuperRank, TLocalRank>(bitmask.data(), pSuperRanks, safeSuperLimit, pLocalRanks, safeLocalLimit, pIndices, pOutRanks, count);
		else
			Internal::GetTwoLevelRanksBatch_Scalar<TSuperRank, TLocalRank>(bitmask.data(), pSuperRanks, pLocalRanks, pIndices, pOutRanks, count);
#endif
	}
}