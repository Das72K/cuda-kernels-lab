# CUDA Kernels Lab

A learning project for NVIDIA internship interviews.

## Setup

- WSL Ubuntu 24.04
- CUDA 13.3
- RTX 4050 Laptop GPU (sm_89, 6 GB)
- CMake, gcc 13

## Build

```bash
cmake -S . -B build && cmake --build build
```

Binaries are placed in `build/bin/`.

## Rules

- **Code changes are designed in a separate mentoring chat.** When I paste a change, apply it
  exactly as given and show me the diff. Do not refactor, "optimize", or rewrite kernels on
  your own.
- **Build/run failures:** if something fails to compile or run, show me the full error and
  explain it briefly. Do not silently fix kernel logic.
- **Benchmarks:** always use Release builds. Never change timing or measurement code unless asked.
- **Free to help with:** README formatting, `.gitignore`, git commits, folder scaffolding.
- **Commits** are authored as Dorin Dascălu.
