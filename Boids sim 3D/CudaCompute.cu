// ============================================================================
//  CudaCompute.cu - GPU boid simulation with spatial grid + flocking
//  CudaCompute.cu - 空間グリッドと群れ行動を使ったGPUボイドシミュレーション
// ============================================================================
//
//  WHAT IS THIS FILE?
//  このファイルは何？
//
//  This file runs on the GPU to simulate thousands of fish swimming together.
//  It uses 3 classic flocking rules (separation, alignment, cohesion) + 
//  obstacle avoidance. A spatial grid makes neighbor lookup fast.
//
//  このファイルはGPUで何千ものボイドを泳がせるシミュレーションです。
//  3つの古典的な群れルール（分離、整列、結合）＋障害物回避を使います。
//  空間グリッドで近隣検索を高速化します。
//
//  CONCEPT: Why GPU?  /  概念：なぜGPU？
//
//  CPU = smart but few cores (like 8 fast workers).
//  GPU = simple but thousands of cores (like 10,000 small workers).
//  For boids, each fish does the SAME calculation → perfect for GPU.
//
//  CPU = 頭良いけどコア少ない（速い作業員8人みたい）
//  GPU = 単純だけど何千ものコア（小さな作業員1万人みたい）
//  ボイドは魚ごとに同じ計算 → GPUにピッタリ
//
// ============================================================================

#define CCCL_IGNORE_MVSC_TRADITIONAL_PREPROCESSOR_WARNING

// ----- IntelliSense fix (editor only, not real build) -----
// エディタの赤線対策（実際のビルドには影響なし）
#ifdef __INTELLISENSE__
#include <device_launch_parameters.h>
#define __CUDACC__
#endif

#define NOMINMAX  // prevent Windows min/max macro / Windowsのmin/maxマクロを防ぐ
#include <Windows.h>
#include "CudaCompute.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>
#include <cmath>
#include <cuda_gl_interop.h>
#include <chrono>


// ============================================================================
//  GPU MEMORY POINTERS  /  GPUメモリへのポインタ
// ============================================================================
//  Prefix "d_" = device (GPU) memory.  /  "d_" = デバイス（GPU）メモリ
//  These pointers live on CPU but POINT TO data on the GPU.
//  ポインタ自体はCPUにあるが、指している先はGPU上のデータ。

// Boid data (position, rotation, velocity) - one float4 per boid
// ボイドのデータ（位置、回転、速度）- ボイド1匹につきfloat4一つ
static float4* d_positions = nullptr;   // xyz = position, w = unused
static float4* d_rotations = nullptr;   // xyz = euler angles (radians)
static float4* d_velocities = nullptr;  // xyz = velocity vector

// Spatial grid data - used to find neighbors fast
// 空間グリッドのデータ - 近くのボイドを素早く探すため
static int* d_cellIds = nullptr;  // which cell each boid is in / 各ボイドが属するセル
static int* d_boidIds = nullptr;  // boid indices (0,1,2...) sorted by cell
static int* d_cellStart = nullptr;  // first boid index in each cell
static int* d_cellEnd = nullptr;  // one past last boid index in each cell

static void* d_tempStorage = nullptr;
static size_t g_tempBytes = 0;
static int* d_cellIdsOut = nullptr;
static int* d_boidIdsOut = nullptr;
static cudaStream_t g_stream = 0;

static cudaEvent_t evStart, evAfterSort, evAfterFlock, evAfterMatrices;

// ============================================================================
//  GRID PARAMETERS  /  グリッドのパラメータ
// ============================================================================
//  "g_" = global (CPU side).  "c_" = constant memory (GPU side, fast read).
//  "g_" = CPU側のグローバル変数。"c_" = GPU側の定数メモリ（超高速読み取り）

// CPU-side copies (used when launching kernels)
// CPU側のコピー（カーネル起動時に使用）
static float g_cellSize;  // size of one cell in world units / 1セルの世界単位サイズ
static float g_worldMin;  // world boundary (assumes cube) / 世界の境界（立方体想定）
static int   g_gridDim;   // cells per axis, e.g. 15 means 15x15x15 grid
static int   g_numCells;  // total cells = gridDim^3

// GPU-side copies (__constant__ = read-only super-fast memory on GPU)
// GPU側のコピー（__constant__ = GPU上の読み取り専用超高速メモリ）
__constant__ float c_cellSize;
__constant__ float c_worldMin;
__constant__ int   c_gridDim;

// Flocking behavior tuning (all on GPU constant memory)
// 群れ行動の調整パラメータ（全てGPU定数メモリ上）
__constant__ float c_perceptionRadius;  // how far each boid sees / ボイドの視野距離
__constant__ float c_separationWeight;  // "don't crowd me" strength / 離れる力
__constant__ float c_alignmentWeight;   // "match heading" strength / 向き合わせる力
__constant__ float c_cohesionWeight;    // "stay together" strength / 集まる力
__constant__ float c_maxSpeed;          // speed cap / 速度上限

// Obstacle avoidance (static obstacles in the world)
// 障害物回避（世界にある静的な障害物）
static float4* d_obstacles = nullptr;   // xyz = center, w = radius
static int g_obstacleCount = 0;
__constant__ int   c_obstacleCount;
__constant__ float c_obstacleWeight;    // how hard to push away / 押し出す強さ

// for matrix drawing
static cudaGraphicsResource* g_resClose = nullptr;
static cudaGraphicsResource* g_resMed = nullptr;
static cudaGraphicsResource* g_resFar = nullptr;
static int* d_countClose = nullptr;
static int* d_countMed = nullptr;
static int* d_countFar = nullptr;

// ============================================================================
//  HELPER: compute which cell a position belongs to
//  ヘルパー：位置からセル番号を計算
// ============================================================================
//
//  EXAMPLE / 例:
//    worldMin = -30, cellSize = 4, gridDim = 15
//    pos = (0, 0, 0) → x = (0 - (-30)) / 4 = 7 → cell (7, 7, 7)
//    flattened index = 7 + 7*15 + 7*15*15 = 1687
//
__device__ int computeCellId(float4 pos) {
    int x = (int)((pos.x - c_worldMin) / c_cellSize);
    int y = (int)((pos.y - c_worldMin) / c_cellSize);
    int z = (int)((pos.z - c_worldMin) / c_cellSize);

    // Clamp so boids outside world still get a valid cell
    // 世界の外にいるボイドも有効なセルを持てるよう制限
    x = max(0, min(c_gridDim - 1, x));
    y = max(0, min(c_gridDim - 1, y));
    z = max(0, min(c_gridDim - 1, z));

    // Flatten 3D coords to 1D index (like indexing a 3D array)
    // 3D座標を1Dインデックスに平坦化（3D配列のアクセスと同じ）
    return x + y * c_gridDim + z * c_gridDim * c_gridDim;
}

// ============================================================================
//  KERNEL: assign each boid to its cell
//  カーネル：各ボイドをセルに割り当てる
// ============================================================================
//  Runs in parallel - one GPU thread per boid.
//  並列実行 - ボイド1匹につきGPUスレッド1つ
//
//  threadIdx.x = thread inside block (0..blockDim.x-1)
//  blockIdx.x  = which block we're in
//  i = unique global ID for this thread (our boid index)
//
//  EXAMPLE / 例:
//    block 0 threads 0-255 → i = 0..255
//    block 1 threads 0-255 → i = 256..511
//
__global__ void assignCellsKernel(float4* pos, int* cellIds, int* boidIds, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;  // guard: more threads than boids / 余分なスレッドを弾く

    cellIds[i] = computeCellId(pos[i]);  // which cell am I in? / 自分のセルは？
    boidIds[i] = i;                       // my own ID / 自分のID
}


// ============================================================================
//  KERNEL: find where each cell's boids start/end in the sorted array
//  カーネル：ソート済み配列で各セルの開始/終了位置を見つける
// ============================================================================
//  After thrust::sort_by_key, boidIds are sorted by cellIds:
//  thrust::sort_by_keyの後、boidIdsはcellIdsでソート済み:
//
//    cellIds:  [ 0, 0, 0, 1, 1, 2, 2, 2, 2 ]
//    boidIds:  [ 5, 2, 9, 1, 7, 0, 3, 4, 8 ]
//                ^        ^        ^
//              cell 0    cell 1  cell 2
//
//    cellStart[0]=0, cellEnd[0]=3   (boids 5,2,9 are in cell 0)
//    cellStart[1]=3, cellEnd[1]=5
//    cellStart[2]=5, cellEnd[2]=9
//
__global__ void findCellBoundsKernel(int* cellIds, int* cellStart, int* cellEnd, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int cell = cellIds[i];

    // First boid → start of its cell
    // 最初のボイド → そのセルの開始位置
    if (i == 0) {
        cellStart[cell] = 0;
    }
    else {
        // If this boid's cell differs from previous, boundary detected
        // 前のボイドとセルが違えば境界を発見
        int prevCell = cellIds[i - 1];
        if (cell != prevCell) {
            cellStart[cell] = i;       // start of new cell / 新しいセルの開始
            cellEnd[prevCell] = i;     // end of previous cell / 前のセルの終了
        }
    }

    // Last boid → end of its cell
    // 最後のボイド → そのセルの終了位置
    if (i == n - 1) {
        cellEnd[cell] = n;
    }
}


// ============================================================================
//  KERNEL: the main flocking calculation
//  カーネル：群れ行動のメイン計算
// ============================================================================
//  For each boid:
//    1. Look at neighbors in nearby cells (3x3x3 = 27 cells)
//    2. Compute 3 steering forces: separation, alignment, cohesion
//    3. Add obstacle avoidance force
//    4. Update velocity, position, rotation
//
//  各ボイドに対して：
//    1. 近くのセル（3x3x3 = 27セル）で近隣ボイドを見る
//    2. 3つの操舵力を計算：分離、整列、結合
//    3. 障害物回避の力を加算
//    4. 速度、位置、回転を更新
//
__global__ void flockingKernel(
    float4* pos, float4* vel, float4* rot,
    int* cellStart, int* cellEnd, int* boidIds,
    float4* obstacles,
    int n, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Load my state into local variables (faster than re-reading global mem)
    // 自分の状態をローカル変数にコピー（グローバルメモリ読み直すより速い）
    float4 myPos = pos[i];
    float4 myVel = vel[i];

    // Accumulators for the 3 flocking forces
    // 3つの群れの力の累積用
    float3 sep = { 0,0,0 };  // separation: push away from neighbors / 分離
    float3 ali = { 0,0,0 };  // alignment: match neighbor velocity / 整列
    float3 coh = { 0,0,0 };  // cohesion: move to center of neighbors / 結合
    int neighborCount = 0;

    float radiusSq = c_perceptionRadius * c_perceptionRadius;

    // Which cell am I in?  /  自分のいるセルは？
    int cx = (int)((myPos.x - c_worldMin) / c_cellSize);
    int cy = (int)((myPos.y - c_worldMin) / c_cellSize);
    int cz = (int)((myPos.z - c_worldMin) / c_cellSize);

    // ------------------------------------------------------------------------
    // Check 3x3x3 cells around me (27 cells total)
    // 周りの3x3x3セル（計27セル）を調べる
    // ------------------------------------------------------------------------
    for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cx + dx, ny = cy + dy, nz = cz + dz;

                // Skip cells outside grid  /  グリッド外のセルはスキップ
                if (nx < 0 || ny < 0 || nz < 0) continue;
                if (nx >= c_gridDim || ny >= c_gridDim || nz >= c_gridDim) continue;

                int cell = nx + ny * c_gridDim + nz * c_gridDim * c_gridDim;
                int start = cellStart[cell];
                if (start < 0) continue;  // empty cell / 空セル
                int end = cellEnd[cell];

                // Loop over all boids in this cell
                // このセルの全ボイドをループ
                for (int k = start; k < end; k++) {
                    int j = boidIds[k];
                    if (j == i) continue;  // skip myself / 自分はスキップ

                    float4 otherPos = pos[j];
                    float rx = otherPos.x - myPos.x;
                    float ry = otherPos.y - myPos.y;
                    float rz = otherPos.z - myPos.z;
                    float distSq = rx * rx + ry * ry + rz * rz;

                    // Too far or exactly on top → ignore
                    // 遠すぎる or 重なり → 無視
                    if (distSq > radiusSq || distSq < 0.0001f) continue;

                    // SEPARATION: push away, stronger when closer
                    // 分離：離れる、近いほど強く
                    sep.x -= rx / distSq;
                    sep.y -= ry / distSq;
                    sep.z -= rz / distSq;

                    // ALIGNMENT: sum neighbors' velocities (averaged later)
                    // 整列：近隣の速度を合計（後で平均）
                    float4 otherVel = vel[j];
                    ali.x += otherVel.x;
                    ali.y += otherVel.y;
                    ali.z += otherVel.z;

                    // COHESION: sum neighbors' positions (averaged later)
                    // 結合：近隣の位置を合計（後で平均）
                    coh.x += otherPos.x;
                    coh.y += otherPos.y;
                    coh.z += otherPos.z;

                    neighborCount++;
                }
            }

    // ------------------------------------------------------------------------
    // Combine the 3 forces into acceleration
    // 3つの力を加速度に合成
    // ------------------------------------------------------------------------
    float3 accel = { 0,0,0 };
    if (neighborCount > 0) {
        float inv = 1.0f / neighborCount;

        // Alignment: (average neighbor velocity) - (my velocity)
        // 整列：(近隣の平均速度) - (自分の速度)
        ali.x = ali.x * inv - myVel.x;
        ali.y = ali.y * inv - myVel.y;
        ali.z = ali.z * inv - myVel.z;

        // Cohesion: (average neighbor position) - (my position)
        // 結合：(近隣の平均位置) - (自分の位置)
        coh.x = coh.x * inv - myPos.x;
        coh.y = coh.y * inv - myPos.y;
        coh.z = coh.z * inv - myPos.z;

        // Weighted sum  /  重み付き合計
        accel.x = sep.x * c_separationWeight + ali.x * c_alignmentWeight + coh.x * c_cohesionWeight;
        accel.y = sep.y * c_separationWeight + ali.y * c_alignmentWeight + coh.y * c_cohesionWeight;
        accel.z = sep.z * c_separationWeight + ali.z * c_alignmentWeight + coh.z * c_cohesionWeight;
    }

    // ------------------------------------------------------------------------
    // OBSTACLE AVOIDANCE: push away from each obstacle if too close
    // 障害物回避：近すぎる障害物から押し離す
    // ------------------------------------------------------------------------
    float3 avoid = { 0,0,0 };
    for (int o = 0; o < c_obstacleCount; o++) {
        float4 obs = obstacles[o];
        float rx = myPos.x - obs.x;
        float ry = myPos.y - obs.y;
        float rz = myPos.z - obs.z;
        float distSq = rx * rx + ry * ry + rz * rz;
        float avoidRadius = obs.w + 2.0f;  // radius + safety buffer / 安全余裕

        if (distSq < avoidRadius * avoidRadius && distSq > 0.0001f) {
            float dist = sqrtf(distSq);
            float inv = 1.0f / dist;
            float push = avoidRadius - dist;  // stronger when closer / 近いほど強く
            avoid.x += rx * inv * push;
            avoid.y += ry * inv * push;
            avoid.z += rz * inv * push;
        }
    }
    accel.x += avoid.x * c_obstacleWeight;
    accel.y += avoid.y * c_obstacleWeight;
    accel.z += avoid.z * c_obstacleWeight;

    // ------------------------------------------------------------------------
    // Physics integration  /  物理積分
    // ------------------------------------------------------------------------
    // velocity += acceleration * dt  (standard Euler integration)
    // 速度 += 加速度 * dt（標準的なオイラー積分）
    myVel.x += accel.x * dt;
    myVel.y += accel.y * dt;
    myVel.z += accel.z * dt;

    // Clamp to max speed  /  最高速度で頭打ち
    float speedSq = myVel.x * myVel.x + myVel.y * myVel.y + myVel.z * myVel.z;
    if (speedSq > c_maxSpeed * c_maxSpeed) {
        float scale = c_maxSpeed / sqrtf(speedSq);
        myVel.x *= scale; myVel.y *= scale; myVel.z *= scale;
    }

    // Enforce min speed so fish never stop
    // 最低速度も強制して魚が止まらないように
    float minSpeed = 5.0f;
    if (speedSq < minSpeed * minSpeed) {
        float scale = minSpeed / (sqrtf(speedSq) + 0.0001f);
        myVel.x *= scale; myVel.y *= scale; myVel.z *= scale;
    }

    // Position update  /  位置の更新
    myPos.x += myVel.x * dt;
    myPos.y += myVel.y * dt;
    myPos.z += myVel.z * dt;

    // World wrap-around: exit one side, enter the other
    // 世界の折り返し：片側から出たら反対側から入る
    float worldSize = c_gridDim * c_cellSize;
    if (myPos.x < c_worldMin) myPos.x += worldSize;
    if (myPos.x > c_worldMin + worldSize) myPos.x -= worldSize;
    if (myPos.y < c_worldMin) myPos.y += worldSize;
    if (myPos.y > c_worldMin + worldSize) myPos.y -= worldSize;
    if (myPos.z < c_worldMin) myPos.z += worldSize;
    if (myPos.z > c_worldMin + worldSize) myPos.z -= worldSize;

    // ------------------------------------------------------------------------
    // Rotation: face the direction of movement
    // 回転：進行方向を向く
    // ------------------------------------------------------------------------
    //  yaw   = rotation around Y (left/right)  /  Y軸回転（左右）
    //  pitch = rotation around X (up/down)     /  X軸回転（上下）
    //
    //  atan2f(x, z) gives angle in XZ plane (horizontal facing)
    //  atan2f(y, horizSpeed) gives pitch (tilt nose up or down)
    //
    //  Guard: if moving almost straight up/down, horizSpeed ~= 0 and
    //  atan2 becomes unstable → keep previous rotation.
    //  ガード：ほぼ垂直移動の時、水平速度≈0で atan2が不安定 → 前の回転を維持
    //
    float speed = sqrtf(myVel.x * myVel.x + myVel.y * myVel.y + myVel.z * myVel.z);
    float inv = 1.0f / (speed + 0.0001f);
    rot[i].x = myVel.x * inv;
    rot[i].y = myVel.y * inv;
    rot[i].z = myVel.z * inv;
    rot[i].w = 0.0f;

    // Write back to global memory  /  グローバルメモリに書き戻し
    pos[i] = myPos;
    vel[i] = myVel;
}

__global__ void buildMatricesKernel(
    const float4* positions, const float4* rotations,
    float3 camPos,
    float* outClose, int* cntClose,
    float* outMed, int* cntMed,
    float* outFar, int* cntFar,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float4 p = positions[i];
    float4 f = rotations[i];   // already a unit forward vector

    // right = normalize(cross(worldUp={0,1,0}, fwd))
    float3 right = { f.z, 0.0f, -f.x };
    float invLen = rsqrtf(right.x * right.x + right.z * right.z + 1e-8f);
    right.x *= invLen; right.z *= invLen;

    // up = cross(fwd, right)
    float3 up = {
        f.y * right.z - f.z * right.y,
        f.z * right.x - f.x * right.z,
        f.x * right.y - f.y * right.x
    };

    // LOD bucket by squared distance
    float dx = p.x - camPos.x, dy = p.y - camPos.y, dz = p.z - camPos.z;
    float distSq = dx * dx + dy * dy + dz * dz;

    float* out; int slot;
    if (distSq < 20.f * 20.f) { slot = atomicAdd(cntClose, 1); out = outClose; }
    else if (distSq < 60.f * 60.f) { slot = atomicAdd(cntMed, 1); out = outMed; }
    else { slot = atomicAdd(cntFar, 1); out = outFar; }

    float* m = out + slot * 16;
    // column 0: right
    m[0] = right.x; m[1] = right.y; m[2] = right.z; m[3] = 0.f;
    // column 1: up
    m[4] = up.x;    m[5] = up.y;    m[6] = up.z;    m[7] = 0.f;
    // column 2: fwd
    m[8] = f.x;     m[9] = f.y;     m[10] = f.z;    m[11] = 0.f;
    // column 3: pos
    m[12] = p.x;    m[13] = p.y;    m[14] = p.z;    m[15] = 1.f;
}


// ============================================================================
//  PUBLIC API  /  公開API
// ============================================================================
//  These are called from the CPU side (Scene_Play.cpp).
//  これらはCPU側（Scene_Play.cpp）から呼ばれます。

// ----------------------------------------------------------------------------
//  BoidsInit: upload initial positions/rotations, generate random velocities
//  BoidsInit: 初期位置・回転をアップロード、ランダム速度を生成
// ----------------------------------------------------------------------------
void BoidsInit(float* positions, float* rotations, int n) {
    size_t size = n * sizeof(float4);

    cudaStreamCreate(&g_stream);

    cudaEventCreate(&evStart);
    cudaEventCreate(&evAfterSort);
    cudaEventCreate(&evAfterFlock);
    cudaEventCreate(&evAfterMatrices);

    // Allocate GPU memory  /  GPUメモリを確保
    cudaMalloc(&d_positions, size);
    cudaMalloc(&d_rotations, size);
    cudaMalloc(&d_velocities, size);

    // Copy initial data from CPU to GPU  /  CPU→GPUへコピー
    cudaMemcpy(d_positions, positions, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rotations, rotations, size, cudaMemcpyHostToDevice);

    // Make random starting velocities so they move from t=0
    // ランダム初速度を作る（最初から動くように）
    float4* initVel = new float4[n];
    for (int i = 0; i < n; i++) {
        initVel[i].x = ((rand() % 200) - 100) / 50.0f;  // range -2..2
        initVel[i].y = ((rand() % 200) - 100) / 50.0f;
        initVel[i].z = ((rand() % 200) - 100) / 50.0f;
        initVel[i].w = 0;
    }
    cudaMemcpy(d_velocities, initVel, size, cudaMemcpyHostToDevice);
    delete[] initVel;
}

// ----------------------------------------------------------------------------
//  SpatialGridInit: set up the grid for fast neighbor search
//  SpatialGridInit: 高速な近隣検索のためのグリッドを設定
// ----------------------------------------------------------------------------
//  EXAMPLE / 例:
//    SpatialGridInit(10000, 4.0f, -30.0f, 30.0f)
//    → world is 60x60x60, cellSize 4 → 15x15x15 = 3375 cells
//    → 世界は60x60x60、セル4 → 15x15x15 = 3375セル
//
void SpatialGridInit(int n, float cellSize, float worldMin, float worldMax) {
    g_cellSize = cellSize;
    g_worldMin = worldMin;
    g_gridDim = (int)ceilf((worldMax - worldMin) / cellSize);
    g_numCells = g_gridDim * g_gridDim * g_gridDim;

    cudaMalloc(&d_cellIds, n * sizeof(int));
    cudaMalloc(&d_boidIds, n * sizeof(int));
    cudaMalloc(&d_cellIdsOut, n * sizeof(int));
    cudaMalloc(&d_boidIdsOut, n * sizeof(int));

    cub::DeviceRadixSort::SortPairs(
        nullptr, g_tempBytes,
        d_cellIds, d_cellIdsOut,
        d_boidIds, d_boidIdsOut,
        n
    );
    cudaMalloc(&d_tempStorage, g_tempBytes);

    cudaMalloc(&d_cellStart, g_numCells * sizeof(int));
    cudaMalloc(&d_cellEnd, g_numCells * sizeof(int));

    // Upload params to GPU constant memory (read-only, fast)
    // パラメータをGPU定数メモリへアップロード（読み取り専用・高速）
    cudaMemcpyToSymbol(c_cellSize, &g_cellSize, sizeof(float));
    cudaMemcpyToSymbol(c_worldMin, &g_worldMin, sizeof(float));
    cudaMemcpyToSymbol(c_gridDim, &g_gridDim, sizeof(int));
}

// ----------------------------------------------------------------------------
//  SpatialGridBuild: rebuild the grid every frame (boids have moved)
//  SpatialGridBuild: 毎フレームでグリッドを再構築（ボイドが動いたので）
// ----------------------------------------------------------------------------
//  Steps / 手順:
//    1. Clear old cell bounds  /  古いセル境界をクリア
//    2. Find each boid's cell  /  各ボイドのセルを計算
//    3. Sort boid IDs by cell  /  セル番号でボイドIDをソート
//    4. Find start/end of each cell in sorted list
//       ソート済みリストの各セルの開始/終了位置を見つける
//
void SpatialGridBuild(int n) {
    int bs = 256, nb = (n + bs - 1) / bs;

    // Step 1: reset cell bounds (-1 = empty, 0xFF as bytes = -1 as int)
    // 手順1：セル境界をリセット（-1 = 空、バイト列0xFFはint型の-1）
    cudaMemset(d_cellStart, 0xFF, g_numCells * sizeof(int));
    cudaMemset(d_cellEnd, 0xFF, g_numCells * sizeof(int));

    // Step 2: each boid computes its cell ID
    // 手順2：各ボイドが自分のセルIDを計算
    assignCellsKernel << <nb, bs >> > (d_positions, d_cellIds, d_boidIds, n);

    cub::DeviceRadixSort::SortPairs(
        d_tempStorage, g_tempBytes,
        d_cellIds, d_cellIdsOut,
        d_boidIds, d_boidIdsOut,
        n, 0, sizeof(int) * 8, g_stream);

    // CUB wrote sorted data into the "Out" buffers. Swap pointers so the rest
    // of the code sees the sorted data in d_cellIds / d_boidIds.
    std::swap(d_cellIds, d_cellIdsOut);
    std::swap(d_boidIds, d_boidIdsOut);

    // Step 4: find where each cell starts/ends
    // 手順4：各セルの開始/終了位置を見つける
    findCellBoundsKernel << <nb, bs >> > (d_cellIds, d_cellStart, d_cellEnd, n);
}

// ----------------------------------------------------------------------------
//  FlockingInit: upload flocking weights (called once at scene start)
//  FlockingInit: 群れ行動のパラメータをアップロード（シーン開始時に1回）
// ----------------------------------------------------------------------------
//  EXAMPLE / 例:
//    FlockingInit(3.0f, 0.5f, 1.0f, 0.5f, 20.0f)
//    → perception 3, separation 0.5, alignment 1, cohesion 0.5, maxSpeed 20
//    → 視野3、分離0.5、整列1、結合0.5、最大速度20
//
void FlockingInit(float perception, float sep, float align, float coh, float maxSpeed) {
    cudaMemcpyToSymbol(c_perceptionRadius, &perception, sizeof(float));
    cudaMemcpyToSymbol(c_separationWeight, &sep, sizeof(float));
    cudaMemcpyToSymbol(c_alignmentWeight, &align, sizeof(float));
    cudaMemcpyToSymbol(c_cohesionWeight, &coh, sizeof(float));
    cudaMemcpyToSymbol(c_maxSpeed, &maxSpeed, sizeof(float));
}

// ----------------------------------------------------------------------------
//  ObstaclesInit: upload obstacle positions and radii
//  ObstaclesInit: 障害物の位置と半径をアップロード
// ----------------------------------------------------------------------------
//  positions is a flat array: [x0,y0,z0, x1,y1,z1, ...]
//  positionsは平坦な配列：[x0,y0,z0, x1,y1,z1, ...]
//
//  EXAMPLE / 例:
//    float pos[] = {0,0,0,  10,5,-5};
//    float rad[] = {3.0f,   2.0f};
//    ObstaclesInit(pos, rad, 2);
//
void ObstaclesInit(float* positions, float* radii, int count) {
    g_obstacleCount = count;
    cudaMemcpyToSymbol(c_obstacleCount, &count, sizeof(int));

    float weight = 20.0f;  // how hard to push / 押し出す強さ
    cudaMemcpyToSymbol(c_obstacleWeight, &weight, sizeof(float));

    cudaMalloc(&d_obstacles, count * sizeof(float4));

    // Pack flat position+radius into float4 on CPU, then upload
    // CPU側で平坦な位置+半径をfloat4にまとめてから送信
    float4* h = new float4[count];
    for (int i = 0; i < count; i++) {
        h[i].x = positions[i * 3 + 0];
        h[i].y = positions[i * 3 + 1];
        h[i].z = positions[i * 3 + 2];
        h[i].w = radii[i];
    }
    cudaMemcpy(d_obstacles, h, count * sizeof(float4), cudaMemcpyHostToDevice);
    delete[] h;
}

// ----------------------------------------------------------------------------
//  BoidsUpdate: runs every frame - the main simulation step
//  BoidsUpdate: 毎フレーム実行 - メインのシミュレーションステップ
// ----------------------------------------------------------------------------
//  dt = frame time in seconds (from GetFrameTime())
//  dt = フレーム時間（秒）（GetFrameTime()から）
//
//  Steps / 手順:
//    1. Rebuild spatial grid  /  空間グリッドを再構築
//    2. Run flocking kernel   /  群れカーネルを実行
//    3. Copy results back to CPU for rendering
//       結果をCPUに戻して描画用に使う
//


//void BoidsUpdate(float* positions, float* rotations, int n, float dt) {
//    SpatialGridBuild(n);
//
//    int bs = 256, nb = (n + bs - 1) / bs;
//    size_t size = n * sizeof(float4);
//
//    flockingKernel << <nb, bs >> > (
//        d_positions, d_velocities, d_rotations,
//        d_cellStart, d_cellEnd, d_boidIds,
//        d_obstacles,
//        n, dt);
//
//    // Copy updated data GPU→CPU for CPU-side matrix building
//    // CPU側の行列計算のためGPU→CPUへコピー
//    cudaMemcpy(positions, d_positions, size, cudaMemcpyDeviceToHost);
//    cudaMemcpy(rotations, d_rotations, size, cudaMemcpyDeviceToHost);
//
//    cudaError_t err = cudaGetLastError();
//    if (err != cudaSuccess) printf("CUDA: %s\n", cudaGetErrorString(err));
//}


// ----------------------------------------------------------------------------
//  Shutdown: free all GPU memory when scene ends
//  シャットダウン：シーン終了時に全GPUメモリを解放
// ----------------------------------------------------------------------------
void BoidsShutdown() {
    cudaFree(d_positions);
    cudaFree(d_rotations);
    cudaFree(d_velocities);
    cudaFree(d_cellIds);
    cudaFree(d_boidIds);
    cudaFree(d_cellStart);
    cudaFree(d_cellEnd);
    cudaFree(d_obstacles);
    cudaFree(d_tempStorage);
    cudaFree(d_cellIdsOut);
    cudaFree(d_boidIdsOut);
}

void* AllocPinned(size_t bytes) {
    void* p = nullptr;
    cudaMallocHost(&p, bytes);
    return p;
}
void FreePinned(void* ptr) {
    cudaFreeHost(ptr);
}

//void BoidsUpdateAsync(float* positions, float* rotations, int n, float dt) {
//    SpatialGridBuild(n);  // you may want this on the stream too; ok for now
//
//    int bs = 256, nb = (n + bs - 1) / bs;
//    size_t size = n * sizeof(float4);
//
//    flockingKernel << <nb, bs, 0, g_stream >> > (
//        d_positions, d_velocities, d_rotations,
//        d_cellStart, d_cellEnd, d_boidIds,
//        d_obstacles, n, dt);
//
//    // async copy — kicks off and returns immediately
//    cudaMemcpyAsync(positions, d_positions, size,
//        cudaMemcpyDeviceToHost, g_stream);
//    cudaMemcpyAsync(rotations, d_rotations, size,
//        cudaMemcpyDeviceToHost, g_stream);
//}

void BoidsRegisterGLBuffers(unsigned int vboClose, unsigned int vboMed,
    unsigned int vboFar, int maxBoids)
{
    cudaGraphicsGLRegisterBuffer(&g_resClose, vboClose,
        cudaGraphicsRegisterFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&g_resMed, vboMed,
        cudaGraphicsRegisterFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&g_resFar, vboFar,
        cudaGraphicsRegisterFlagsWriteDiscard);

    cudaMalloc(&d_countClose, sizeof(int));
    cudaMalloc(&d_countMed, sizeof(int));
    cudaMalloc(&d_countFar, sizeof(int));
}

void BoidsUnregisterGLBuffers() {
    if (g_resClose) cudaGraphicsUnregisterResource(g_resClose);
    if (g_resMed)   cudaGraphicsUnregisterResource(g_resMed);
    if (g_resFar)   cudaGraphicsUnregisterResource(g_resFar);
    cudaFree(d_countClose); cudaFree(d_countMed); cudaFree(d_countFar);
}

void BoidsUpdateGPUOnly(int n, float dt) {
    cudaEventRecord(evStart, g_stream);

    SpatialGridBuild(n);
    cudaEventRecord(evAfterSort, g_stream);

    int bs = 256, nb = (n + bs - 1) / bs;
    flockingKernel << <nb, bs, 0, g_stream >> > (
        d_positions, d_velocities, d_rotations,
        d_cellStart, d_cellEnd, d_boidIds,
        d_obstacles, n, dt);
    cudaEventRecord(evAfterFlock, g_stream);
}

void BoidsBuildMatricesGPU(float camX, float camY, float camZ, int n,
    int* outClose, int* outMed, int* outFar)
{
    // zero counters
    cudaMemsetAsync(d_countClose, 0, sizeof(int), g_stream);
    cudaMemsetAsync(d_countMed, 0, sizeof(int), g_stream);
    cudaMemsetAsync(d_countFar, 0, sizeof(int), g_stream);

    // map VBOs → CUDA pointers
    cudaGraphicsResource* res[3] = { g_resClose, g_resMed, g_resFar };
    cudaGraphicsMapResources(3, res, g_stream);

    float* dClose, * dMed, * dFar;
    size_t bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&dClose, &bytes, g_resClose);
    cudaGraphicsResourceGetMappedPointer((void**)&dMed, &bytes, g_resMed);
    cudaGraphicsResourceGetMappedPointer((void**)&dFar, &bytes, g_resFar);

    float3 cam = { camX, camY, camZ };
    int bs = 256, nb = (n + bs - 1) / bs;
    buildMatricesKernel << <nb, bs, 0, g_stream >> > (
        d_positions, d_rotations, cam,
        dClose, d_countClose,
        dMed, d_countMed,
        dFar, d_countFar,
        n);

    // unmap (implicit sync → kernel is done)
    cudaGraphicsUnmapResources(3, res, g_stream);

    // read counters (tiny copies)
    cudaMemcpyAsync(outClose, d_countClose, sizeof(int),
        cudaMemcpyDeviceToHost, g_stream);
    cudaMemcpyAsync(outMed, d_countMed, sizeof(int),
        cudaMemcpyDeviceToHost, g_stream);
    cudaMemcpyAsync(outFar, d_countFar, sizeof(int),
        cudaMemcpyDeviceToHost, g_stream);
    cudaEventRecord(evAfterMatrices, g_stream);
    cudaStreamSynchronize(g_stream);
}


void BoidsGetTimings(float* sortMs, float* flockMs, float* matricesMs) {
    cudaEventSynchronize(evAfterMatrices);  // make sure all events done
    cudaEventElapsedTime(sortMs, evStart, evAfterSort);
    cudaEventElapsedTime(flockMs, evAfterSort, evAfterFlock);
    cudaEventElapsedTime(matricesMs, evAfterFlock, evAfterMatrices);
}


void BoidsTimedSteps(int n, float dt, float* sortMs, float* flockMs) {
    auto t0 = std::chrono::high_resolution_clock::now();
    SpatialGridBuild(n);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    int bs = 256, nb = (n + bs - 1) / bs;
    flockingKernel << <nb, bs, 0, g_stream >> > (
        d_positions, d_velocities, d_rotations,
        d_cellStart, d_cellEnd, d_boidIds,
        d_obstacles, n, dt);
    cudaStreamSynchronize(g_stream);
    auto t2 = std::chrono::high_resolution_clock::now();

    *sortMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
    *flockMs = std::chrono::duration<float, std::milli>(t2 - t1).count();
}
