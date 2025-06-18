# Parallel Image Convolution Benchmark Tool

This project is a parallel image convolution benchmarking tool that features:

- Support for various convolution kernels (blur, sobel, sharpen, laplacian, etc.)
- Configurable block sizes, splitting strategies, OpenMP scheduling modes, and thread counts
- False sharing detection and CPU core affinity binding (P-core only or all logical processors)
- Optional intermediate image saving and detailed performance output in CSV

---

## Requirements

- OS: Windows 10 or later
- Compiler: MSVC with C++17 support (Visual Studio recommended)
- Libraries:
  - [OpenCV](https://opencv.org/) (Recommended version: 4.x)
  - OpenMP (included with Visual Studio by default)

Make sure your OpenCV include/lib paths are properly configured.

---

## Project Structure

| File         | Description |
|--------------|-------------|
| `main.cpp`   | Main program that handles input, processing, and benchmarking |
| `modules.h`  | Header file declaring custom classes and data structures |
| `modules.cpp`| Implementation of image loading, normalization, convolution, integration, and output |

---

## Execution Flow

1. Prompt to save intermediate images (optional).
2. Input grayscale image file path.
3. Select a convolution kernel or run all kernels.
4. Automatically benchmark using combinations of:
   - **Splitting strategies**: `rowWise`, `blockWise`
   - **Scheduling modes**: `static`, `dynamic`, `guided`
   - **Block sizes**: 4 to 512 (doubling)
   - **Thread counts**: 1, 2, 4, 8, 16
   - **With/without padding**
   - **CPU Affinity**: bind to performance cores or all cores
5. Repeats the process for 3 trials and saves results to `results{N}/comparison_results.csv`.

---

## Output

- CSV includes:
  - Kernel name, kernel size, block size
  - Execution time (average, min, max, std dev)
  - Speedup and efficiency
  - False sharing count and rate
- Optionally saved processed images (if enabled)

---

## 📊 False Sharing Detection

When padding is disabled, false sharing is tracked by monitoring ownership of 64-byte cache lines using `std::atomic<int>`. If thread ownership switches, it's counted as a false sharing event.

---

## Parallel Optimization Features

- Convolution is parallelized using OpenMP
- Uses `SetProcessAffinityMask` to bind processes to P-cores or all cores
- Allows benchmarking with various thread counts and scheduling modes for performance comparison

---

## Notes

- Only 8-bit grayscale images are supported
- Output images are saved as PNG with pixel values normalized (0–1) and scaled by 255
- Make sure the current directory is writable before running

---

## Use Cases

This tool is ideal for:

- Parallel computing / OpenMP hands-on labs
- Memory/cache-level performance and false sharing studies
- High-performance image preprocessing development

