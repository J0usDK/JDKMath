#pragma once

#if defined(_MSC_VER)
	#include <intrin.h>
#else
	#include <x86intrin.h>
	#include <cpuid.h>
#endif

namespace JDK::Math
{
	struct SCpuFeatures
	{
		bool bHasPopcnt				= false;
		bool bHasAVX512F			= false;
		bool bHasAVX512VL			= false;
		bool bHasAVX512BW			= false;
		bool bHasAVX512_VPOPCNTDQ	= false;
	};

	namespace Internal
	{
		[[nodiscard]] inline SCpuFeatures Detect() noexcept
		{
			SCpuFeatures features;

#if defined(_MSC_VER)
			int info[4];
			__cpuid(info, 1);
			features.bHasPopcnt = (info[2] & (1 << 23)) != 0;

			__cpuidex(info, 7, 0);
			features.bHasAVX512F = (info[1] & (1 << 16)) != 0;
			features.bHasAVX512BW = (info[1] & (1 << 30)) != 0;
			features.bHasAVX512VL = (info[1] & (1 << 31)) != 0;
			features.bHasAVX512_VPOPCNTDQ = (info[2] & (1 << 14)) != 0;
#else
			unsigned int eax, ebx, ecx, edx;
			if (__get_cpuid(1, &eax, &ebx, &ecx, &edx))
				features.bHasPopcnt = (ecx & (1 << 23)) != 0;

			if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
			{
				features.bHasAVX512F = (ebx & (1 << 16)) != 0;
				features.bHasAVX512BW = (ebx & (1 << 30)) != 0;
				features.bHasAVX512VL = (ebx & (1 << 31)) != 0;
				features.bHasAVX512_VPOPCNTDQ = (ecx & (1 << 14)) != 0;
			}
#endif

			if (features.bHasAVX512F)
			{
				bool bOSSupportsAVX512 = false;
#if defined(_MSC_VER)
				unsigned long long xcr0 = _xgetbv(0);
#else
				unsigned int eax_xcr, edx_xcr;
				__asm__ volatile ("xgetbv" : "=a"(eax_xcr), "=d"(edx_xcr) : "c"(0));
				unsigned long long xcr0 = (static_cast<unsigned long long>(edx_xcr) << 32) | eax_xcr;
#endif
				bOSSupportsAVX512 = (xcr0 & 0xE6) == 0xE6;

				if (!bOSSupportsAVX512)
				{
					features.bHasAVX512F = false;
					features.bHasAVX512VL = false;
					features.bHasAVX512_VPOPCNTDQ = false;
				}
			}

			return features;
		}
	}

	[[nodiscard]] inline const SCpuFeatures& GetCpuFeatures() noexcept
	{
		static const SCpuFeatures cpuFeatures = Internal::Detect();
		return cpuFeatures;
	}
}