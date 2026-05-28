<h1 align="center">3D Boids Simulation</h1>

<p align="center">
  GPU-accelerated 3D flocking of <b>200,000+ boids at 60 FPS</b> on a single RTX 2060<br>
  Built with <b>CUDA</b>, <b>CUDA–OpenGL interop</b>, and <b>raylib</b>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white" alt="C++17">
  <img src="https://img.shields.io/badge/CUDA-13.2-76B900?logo=nvidia&logoColor=white" alt="CUDA 13.2">
  <img src="https://img.shields.io/badge/raylib-5.x-white?logo=raylib&logoColor=black" alt="raylib">
  <img src="https://img.shields.io/badge/OpenGL-interop-5586A4?logo=opengl&logoColor=white" alt="OpenGL">
  <img src="https://img.shields.io/badge/Platform-Windows%20x64-0078D6?logo=windows&logoColor=white" alt="Windows x64">
</p>

<p align="center">
  <img src="demo.gif" alt="3D Boids Simulation demo" width="820"><br>
  <sub>Live capture — the clip is large, so it may take a moment to load.</sub>
</p>

---

## Overview

A real-time 3D boids (flocking) simulation that scales from a few thousand agents to
**hundreds of thousands** while holding 60 FPS. Classic flocking rules — separation,
alignment, cohesion, plus obstacle avoidance — run as a CUDA kernel over a 3D spatial
grid, and transform matrices are written **directly into OpenGL vertex buffers** so the
CPU never touches per-boid data on the hot path.

The project started as a CPU prototype (~5,000 boids) and was optimized step by step,
profiling each change with **Nsight Systems** rather than guessing — ending at
**200,000+ boids at 60 FPS**, a **40×+** improvement.

## Highlights

- **CUDA flocking kernel** — separation / alignment / cohesion + spherical obstacle avoidance.
- **3D spatial grid** — each boid checks only its 27 neighboring cells instead of every other boid.
- **CUDA–OpenGL interop** — CUDA writes instance matrices straight into GL VBOs; zero CPU roundtrip.
- **Level of detail (LOD)** — 3 fish mesh tiers selected per boid by camera distance.
- **CUB radix sort** — persistent temp buffer, no per-frame GPU allocations.
- **Pinned host memory & async double buffering** — overlap compute with transfer.
- **Scene system** — menu / play scenes managed by a lightweight scene manager.

## Performance

Measured on an **NVIDIA RTX 2060**, raylib (C++), CUDA 13.2.

| Stage | Boids   | FPS | Technique added |
|:-----:|:-------:|:---:|-----------------|
| 0 | 5,000   | 80 | CPU-only, `DrawMeshInstanced` |
| 1 | 10,000  | 80 | CUDA flocking kernel + spatial grid |
| 2 | 13,000  | 60 | CUB radix sort (replaced Thrust) |
| 3 | 16,000  | 60 | Pinned host memory (`cudaMallocHost`) |
| 4 | 20,000  | 64 | Async double buffering |
| 5 | 35,000  | 60 | LOD system (3 mesh tiers) |
| 6 | 52,000  | 60 | CUDA–OpenGL interop |
| 7 | 150,000 | 60 | Release build (was stuck in Debug) |
| 8 | 200,000 | 60 | Tuned world / grid / perception parameters |

> **Result:** from **5,000 boids (CPU)** to **200,000+ boids (GPU)** at 60 FPS — peaks
> around **260,000**. Every step was driven by measurement, not guesswork.

## How It Works

Each optimization stage and what it changed:

**CPU → GPU (Stage 0→1).** Moved flocking logic to a CUDA kernel and added a 3D spatial
grid, so each boid only checks 27 nearby cells instead of every other boid.

**CUB radix sort (1→2).** `thrust::sort_by_key` was allocating and freeing GPU memory
every frame — a hidden bottleneck. Replaced with `cub::DeviceRadixSort` using a
persistent temp buffer allocated once.

**Pinned memory (2→3).** Device-to-host `cudaMemcpy` was the frame bottleneck. Switching
host arrays to page-locked memory (`cudaMallocHost`) made the copy 2–3× faster.

**Async double buffering (3→4).** Compute frame *N+1* while drawing frame *N*, using
`cudaMemcpyAsync` and two buffer sets on a dedicated stream so CPU matrix building
overlaps with GPU transfer.

**LOD system (4→5).** Three fish mesh variants (480 / 144 / 50 faces) chosen per boid by
distance to camera — distant boids draw the cheap mesh.

**CUDA–OpenGL interop (5→6).** CUDA writes transform matrices directly into GL vertex
buffers via `cudaGraphicsGLRegisterBuffer`, sharing one VRAM buffer between CUDA and
OpenGL — eliminating the CPU roundtrip entirely.

**Release build (6→7).** Debug builds compile CUDA with `-G`, disabling GPU optimizations
(5–10× slower). The Release `.exe` also had to be re-routed to the NVIDIA GPU in Windows
Graphics Settings (it was defaulting to the AMD integrated GPU).

**Parameter tuning (7→8).** Adjusted world size, grid cell size, and perception radius to
lower boid density per cell — fewer neighbor checks per frame, same visual behavior.

## Architecture

```
Boids sim 3D/
├─ main.cpp              # Entry point
├─ Game.{h,cpp}          # Game loop: Start / Update / Render
├─ Scene_Base.h          # Scene interface
├─ Scene_Manager.h       # Scene stack / transitions
├─ Scene_Menu.{h,cpp}    # Main menu
├─ Scene_Play.{h,cpp}    # Simulation scene
├─ CudaCompute.{h,cu}    # GPU simulation: grid, flocking, sort, GL interop
├─ Globals.h             # Window / global config
├─ rlights.h             # raylib lighting helper
└─ Assets/               # Fish LOD meshes, textures, instanced shaders
```

The public GPU API (`CudaCompute.h`) deliberately exposes only plain C/C++ types — no
CUDA types leak into headers, which avoids conflicts with raylib (e.g. `float3`
redefinition).

## Tech Stack

- **Language:** C++17
- **Rendering:** raylib + OpenGL (instanced rendering, custom lighting shaders)
- **Compute:** CUDA 13.2 (CUB radix sort, spatial grid, CUDA–OpenGL interop)
- **Profiling:** NVIDIA Nsight Systems
- **Toolchain:** Visual Studio 2022 (x64)

## Requirements & Build

> This repository contains **source and assets only** — no prebuilt binaries.

To build it yourself you'll need:

- **Visual Studio 2022** with the Desktop C++ workload
- **CUDA Toolkit 13.2** (with Visual Studio integration)
- An **NVIDIA GPU** (developed and tuned on an RTX 2060)
- **raylib** (resolved via vcpkg in the original setup)

Open `Boids sim 3D.slnx` in Visual Studio, select the **x64 / Release** configuration,
and build. Running in **Release** is important — Debug disables CUDA optimizations and is
dramatically slower.

## Key Learnings

- **Profile, don't guess.** Nsight Systems revealed the real bottlenecks every time.
- **Watch the GPU your display actually uses.** CUDA was running on the NVIDIA card while
  the display ran on the AMD integrated GPU — the cross-GPU copy stalls were invisible
  but devastating.
- **Beware hidden allocations.** `cudaFree` bars in the profile timeline traced back to
  Thrust's internal per-frame allocations.
