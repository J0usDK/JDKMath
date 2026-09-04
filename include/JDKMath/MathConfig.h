#pragma once

// ============================================================================
// JDK MATH DISPATCH CONFIGURATION
// ============================================================================
// Controls compile-time dispatching for SIMD and hardware-specific instructions.
// By overriding these macros globally via your build system (e.g., CMake flags 
// or Visual Studio Preprocessor Definitions), you can strip runtime CPUID 
// branches and force specific execution paths for known target architectures.
// 
// Valid configuration values:
//  -1 : Automatic (Resolves to runtime CPUID detection and branching)
//   0 : Force Software/Scalar fallback (Strips SIMD/Hardware instructions)
//   1 : Force Hardware/SIMD path (Bypasses checks, requires hardware support)
// ============================================================================

// Dispatches 64-bit population count (POPCNT)
#ifndef JDK_MATH_DISPATCH_POPCNT
	#define JDK_MATH_DISPATCH_POPCNT -1
#endif

// Dispatches the prefix sum rank table building algorithm
#ifndef JDK_MATH_DISPATCH_AVX512_RANK_BUILD
	#define JDK_MATH_DISPATCH_AVX512_RANK_BUILD -1
#endif

// Dispatches the masked batch processing for tile rank queries
#ifndef JDK_MATH_DISPATCH_AVX512_RANK_GETBATCH
	#define JDK_MATH_DISPATCH_AVX512_RANK_GETBATCH -1
#endif

// Defines the threshold of using two-level instead of flat rank tables (in bytes)
#ifndef JDK_MATH_TWOLEVEL_RANK_THRESHOLD_BYTES
	#define JDK_MATH_TWOLEVEL_RANK_THRESHOLD_BYTES 32 * 1024 * 1024
#endif