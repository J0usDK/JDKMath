#pragma once

// ============================================================================
// JDK MATH ASSERTION SYSTEM
// ============================================================================
// Provides a customizable assertion wrapper to seamlessly integrate with external macros.
// 
// To use a custom macro (e.g., CRY_ASSERT_MESSAGE, checkf), define 
// JDK_MATH_CUSTOM_ASSERT globally before including any JDK Math headers.
//
// Example implementation:
// #define JDK_MATH_CUSTOM_ASSERT(condition, message) CRY_ASSERT_MESSAGE(condition, message)
// ============================================================================

#ifdef JDK_MATH_CUSTOM_ASSERT
	// Proxies the validation to the user-defined assertion macro
	#define JDK_MATH_ASSERT(condition, message) JDK_MATH_CUSTOM_ASSERT(condition, message)
#else
	// Fallback to the standard C++ library assert. 
	// The logical AND ensures the string literal is evaluated and displayed 
	// in the standard output if the assertion fails.
	#include <cassert>
	#define JDK_MATH_ASSERT(condition, message) assert((condition) && (message))
#endif