# LLM.c Kernels in PTX and CUDA

## Setup

```bash
./scripts/download_starter_pack.sh
```

## Compilation

### PTX Version

```bash
nvcc --use_fast_math -std=c++17 -O3 infer_gpt2_ptx.cu -o infer_gpt2_ptx -lcuda -arch=sm_80
```

### CUDA Version

```bash
nvcc --use_fast_math -std=c++17 -O3 infer_gpt2.cu -o infer_gpt2 -lcuda -arch=sm_80
```

## Performance

The PTX version currently runs approximately 1.1 times faster than the CUDA version on my system.
(e.g., CUDA: 1474ms, PTX: 1340ms). Note that these timings can vary.

## TODO

- A detailed step-by-step guide on writing these kernels.
- In-depth explanations of the code will be added to the `docs/` directory.
