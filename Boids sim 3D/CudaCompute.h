#pragma once
// ============================================================================
//  CudaCompute.h - public API for the GPU boid simulation
//  CudaCompute.h - GPUボイドシミュレーションの公開API
// ============================================================================
//  Only plain C/C++ types here - no CUDA types in this header.
//  Why? CUDA headers conflict with raylib (e.g. float3 redefinition).
//
//  ここにはC/C++の普通の型だけ - CUDA型はヘッダに入れない。
//  理由：CUDAヘッダがraylibと衝突する（例：float3の再定義）。
// ============================================================================



// ----------------------------------------------------------------------------
//  Lifecycle / ライフサイクル
// ----------------------------------------------------------------------------

// Allocate GPU buffers and upload initial boid state.
// Call once at the start of the scene.
// GPUバッファを確保し初期状態をアップロード。シーン開始時に1回呼ぶ。
//
// positions, rotations: flat float arrays, 4 floats per boid (x, y, z, unused)
// positions, rotations: 平坦なfloat配列、ボイド1匹につき4float
void BoidsInit(float* positions, float* rotations, int n);


// Legacy not used anymore
// Run one simulation step. Call every frame.
// dt = frame time in seconds.
// 1ステップ実行。毎フレーム呼ぶ。dtはフレーム時間（秒）。
//void BoidsUpdate(float* positions, float* rotations, int n, float dt);

// Free all GPU memory. Call once at scene end.
// 全GPUメモリを解放。シーン終了時に1回呼ぶ。
void BoidsShutdown();


// ----------------------------------------------------------------------------
//  Spatial grid / 空間グリッド
// ----------------------------------------------------------------------------

// Set up the grid used for fast neighbor search.
// 高速な近隣検索のためのグリッドを設定。
//
// n        = number of boids
// cellSize = one cell's edge length (make this ~= perceptionRadius)
//            1セルの辺の長さ（視野半径くらいがベスト）
// worldMin/worldMax = world bounds (world is a cube)
//                    世界の範囲（立方体を想定）
void SpatialGridInit(int n, float cellSize, float worldMin, float worldMax);

// Rebuild the grid (called internally by BoidsUpdate every frame).
// グリッドを再構築（通常はBoidsUpdate内部で毎フレーム呼ばれる）。
void SpatialGridBuild(int n);


// ----------------------------------------------------------------------------
//  Flocking behavior tuning / 群れ行動のチューニング
// ----------------------------------------------------------------------------

// Set flocking weights. Call once after BoidsInit.
// 群れ行動の重みを設定。BoidsInitの後に1回呼ぶ。
//
// perception = how far each boid sees (world units) / 視野距離
// sep        = separation weight (don't crowd)     / 分離の重み
// align      = alignment weight (match heading)    / 整列の重み
// coh        = cohesion weight (stay together)     / 結合の重み
// maxSpeed   = speed cap                           / 速度上限
//
// EXAMPLE / 例:
//   FlockingInit(3.0f, 0.5f, 1.0f, 0.5f, 20.0f);
void FlockingInit(float perception, float sep, float align, float coh, float maxSpeed);


// ----------------------------------------------------------------------------
//  Obstacles / 障害物
// ----------------------------------------------------------------------------

// Upload a list of spherical obstacles.
// 球状の障害物リストをアップロード。
//
// positions = flat array [x0,y0,z0, x1,y1,z1, ...]
// radii     = one float per obstacle
// count     = number of obstacles
//
// EXAMPLE / 例:
//   float pos[] = {0,0,0,  10,5,-5};
//   float rad[] = {3.0f,   2.0f};
//   ObstaclesInit(pos, rad, 2);
void ObstaclesInit(float* positions, float* radii, int count);
void* AllocPinned(size_t bytes);
void  FreePinned(void* ptr);

// Legacy not used anymore
//void BoidsUpdateAsync(float* positions, float* rotations, int n, float dt);
//void BoidsWaitComplete();

// Register 3 GL buffer IDs with CUDA (one per LOD). Call once.
void BoidsRegisterGLBuffers(unsigned int vboClose,
    unsigned int vboMed,
    unsigned int vboFar,
    int maxBoids);

// Build matrices on GPU into the registered VBOs. Returns counts.
void BoidsBuildMatricesGPU(float camX, float camY, float camZ,
    int n,
    int* outCountClose,
    int* outCountMed,
    int* outCountFar);

// Cleanup.
void BoidsUnregisterGLBuffers();

// Replaces BoidsUpdateAsync for the interop path (no CPU copy).
void BoidsUpdateGPUOnly(int n, float dt);


void BoidsGetTimings(float* sortMs, float* flockMs, float* matricesMs);
void BoidsTimedSteps(int n, float dt, float* sortMs, float* flockMs);