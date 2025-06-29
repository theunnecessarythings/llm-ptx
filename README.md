# LLM.c Kernels in PTX

This project is my attempt at handwriting kernels in CUDA Parallel Thread Execution (PTX) for a simple LLM (Large Language Model). I used the GPT-2 model implementation
from [`llm.c`](https://github.com/karpathy/llm-c) by Andrej Karpathy as a starting point.

## What is PTX?

PTX is a low-level virtual instruction set architecture (ISA) for NVIDIA GPUs. It acts as an intermediary language between high-level languages like CUDA and the GPU's native machine code. By writing kernels directly in PTX, you can achieve a finer degree of control over the GPU's resources, potentially leading to significant performance gains.

## Project Structure

The project is organized as follows:

- `main.cu`: The main application file, which can be compiled to use either the PTX or CUDA kernels.
- `kernels_ptx.h`: The header file for the PTX kernels.
- `kernels_cuda.cuh`: The header file for the CUDA kernels.
- `ptx/`: This directory contains the PTX assembly code for the various kernels used in the LLM.
- `llmc/`: This directory contains common code for the LLM, such as the tokenizer and utility functions.
- `scripts/`: This directory contains scripts for downloading the necessary data and models.
- `data/`: This directory contains the datasets for the LLM.

## Setup

To get started, you'll need to download the necessary data and models. You can do this by running the following script:

```bash
./scripts/download_starter_pack.sh
```

This will download the GPT-2 model weights and the TinyShakespeare and Hellaswag datasets.

## Compilation

You can compile the project to use either the PTX or CUDA kernels.

### PTX Version

To compile the PTX version, run the following command:

```bash
nvcc --use_fast_math -std=c++17 -O3 main.cu -o main_ptx -lcuda -arch=sm_80 -DPTX
./main_ptx
```

The `-DPTX` flag tells the compiler to use the PTX kernels.

### CUDA Version

To compile the CUDA version, run the following command:

```bash
nvcc --use_fast_math -std=c++17 -O3 main.cu -o main_cuda -lcuda -arch=sm_80
./main_cuda
```

## Performance

The PTX version currently runs approximately 1.1 times faster than the CUDA version on my system.
(e.g., CUDA: 1474ms, PTX: 1340ms). Note that these timings can vary.

## TODO

- A detailed step-by-step guide on writing these kernels.
- In-depth explanations of the code will be added to the `docs/` directory.
