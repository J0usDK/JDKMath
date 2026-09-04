# JDKMath

JDKMath is high-performance, C++17 header-only library for bit manipulation and rank indexing. It provides SIMD-accelerated rank-table construction and batch queries, with AVX-512 acceleration and scalar fallbacks.

Designed for maximum throughput using SIMD optimizations (**AVX-512** with runtime dispatch / compile-time overrides) alongside scalar fallbacks.

---

## Features

* **Header-Only & Zero Dependency:** Easily drop into any CMake or MSVC/GCC/Clang project.
* **SIMD Accelerated:** Utilizes AVX-512 instructions (`AVX512F`, `AVX512VL`, `AVX512BW`, `AVX512_VPOPCNTDQ`) for both table construction and batch queries.
* **Automatic CPU Dispatch:** Detects hardware support via CPUID/XGETBV at runtime with configuration macros to force compile-time code paths.
* **Multiple Rank Table Architectures:**
  * `CFlatRankTable`: Flat cumulative rank dictionary optimized for raw $O(1)$ query throughput.
  * `CTwoLevelRankTable`: Two-level hierarchical rank dictionary (L1 super-ranks + L2 local offsets) providing up to a ~50% reduction in memory usage.
  * `CUniversalRankTable`: A dynamic rank dictionary wrapper that automatically chooses the optimal storage width (`uint8_t` through `uint64_t`) and architecture based on the input bitmask size.
* **Batch Lookups:** Vectorized `GetRanksBatch` processing up to 8 indices simultaneously with SIMD gather instructions.
* **Extensible Assertions:** Overridable assertion system via `JDK_MATH_CUSTOM_ASSERT` for seamless game engine integration (Unreal Engine, CryEngine, etc.).

---

## Performance & Architecture

| Architecture | Query Time | Memory Overhead (per 64-bit word) | Best Used For |
| :--- | :--- | :--- | :--- |
| **Flat Table** | Fastest ($O(1)$) | `sizeof(TRank)` bytes | Maximum speed queries, small-to-medium bitmasks |
| **Two-Level Table** | Near-Flat ($O(1)$) | `sizeof(TLocalRank) + sizeof(TSuperRank) / 32` bytes | Large bitmasks where cache efficiency and memory footprint matter |
| **Universal Table** | Adaptive dispatch | Adaptive | Bitmasks with variable runtime sizes |

---

## Directory Structure

```text
JDKMath/
├── CMakeLists.txt
├── README.md
└── include/
    └── JDKMath/
        ├── BitMath.h
        ├── BitmaskRank.h
        ├── CpuFeatures.h
        ├── FlatRankTable.h
        ├── MathAssert.h
        ├── MathConfig.h
        ├── MathGeneral.h
        ├── TwoLevelRankTable.h
        └── UniversalRankTable.h
```

---

## Integration

### CMake

Add `JDKMath` as a subdirectory or link it via `FetchContent`:

```cmake
add_subdirectory(path/to/JDKMath)
target_link_libraries(YourProject PRIVATE JDKMath)
```

The library exposes standard include directories:

```cpp
#include <JDKMath/UniversalRankTable.h>
```

---

## Configuration

Predefined preprocessor switches can be passed via CMake to strip runtime dynamic dispatch checks:

| Macro | Values | Description |
| :--- | :--- | :--- |
| `JDK_MATH_DISPATCH_POPCNT` | `-1` (Auto) / `0` (Scalar) / `1` (HW) | Controls popcount instruction path. |
| `JDK_MATH_DISPATCH_AVX512_RANK_BUILD` | `-1` (Auto) / `0` (Scalar) / `1` (AVX512) | Rank table build path dispatch. |
| `JDK_MATH_DISPATCH_AVX512_RANK_GETBATCH` | `-1` (Auto) / `0` (Scalar) / `1` (AVX512) | Batch rank gather dispatch. |
| `JDK_MATH_TWOLEVEL_RANK_THRESHOLD_BYTES` | Bytes (`default: 32MB`) | Memory threshold above which `CUniversalRankTable` uses Two-Level tables. |
| `JDK_MATH_CUSTOM_ASSERT(cond, msg)` | Macro expression | Custom engine assertion hook (defaults to standard `assert`). |