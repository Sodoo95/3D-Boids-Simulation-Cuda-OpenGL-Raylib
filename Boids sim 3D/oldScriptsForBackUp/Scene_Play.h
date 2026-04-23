#pragma once

// ============================================================================
//  Scene_Play.h - the main simulation scene
//  Scene_Play.h - メインのシミュレーションシーン
// ============================================================================
//  Holds all CPU-side state: boid arrays, camera, model, obstacles.
//  GPU work is delegated to CudaCompute.
//
//  CPU側の状態を全て保持：ボイド配列、カメラ、モデル、障害物。
//  GPUの処理はCudaComputeに委譲。
// ============================================================================

#include "Scene_Base.h"
#include <string>


// ----------------------------------------------------------------------------
//  Compile-time constants / コンパイル時定数
// ----------------------------------------------------------------------------

// Number of fish to simulate. Keep in sync with GPU memory allocations.
// シミュレーションする魚の数。GPUメモリの確保と一致させる。
const int MAX_BOIDS = 200000;


// Number of obstacles in the world.
// 世界にある障害物の数。
const int NUM_OBSTACLES = 30;


// ----------------------------------------------------------------------------
//  Obstacle struct - sphere in world space
//  障害物構造体 - ワールド空間の球
// ----------------------------------------------------------------------------
struct Obstacle {
    Vector3 pos;    // center position / 中心位置
    float radius;   // sphere radius   / 球の半径
};


// ============================================================================
//  Scene_Play class  /  Scene_Playクラス
// ============================================================================
class Scene_Play : public Scene_Base
{
    std::string boidCount;
    Vector3 pointLightPos{ 0,50,0 };
    Vector3 pointLightDir{ 0,0,0 };
    // --------------------------------------------------------------------
    // Rendering resources  /  描画リソース
    // --------------------------------------------------------------------
    Camera3D   camera;            // 3D camera / 3Dカメラ
    Model      fishModelClose;         // fish mesh / 魚のメッシュ
    Model      fishModelMed;         // fish mesh / 魚のメッシュ
    Model      fishModelFar;         // fish mesh / 魚のメッシュ
    Material   instanceMaterial;  // material for instanced draw / インスタンス描画用
    Texture2D  fishTexture;       // fish texture / 魚のテクスチャ
    Shader     shader;            // custom instancing + lighting shader / カスタムシェーダ

    // --------------------------------------------------------------------
    // Boid data (CPU copies - GPU copies are in CudaCompute.cu)
    // ボイドデータ（CPUコピー - GPUコピーはCudaCompute.cuに）
    // --------------------------------------------------------------------
    int     boidID[MAX_BOIDS];          // reserved / 予約
    //Vector4* boidPositions[2] = { nullptr, nullptr };   // x,y,z,unused / 位置
    //Vector4* boidRotations[2] = { nullptr, nullptr };   // pitch,yaw,roll,unused / オイラー角
    //int bufIdx = 0;
    //Matrix MtransClose[MAX_BOIDS];
    //Matrix MtransMed[MAX_BOIDS];
    //Matrix MtransFar[MAX_BOIDS]; // per-boid transform matrix / ボイド毎の変換行列
    //Matrix  MboidRotations[MAX_BOIDS];  // reserved / 予約

    unsigned int vboClose = 0, vboMed = 0, vboFar = 0;
    int countClose = 0, countMed = 0, countFar = 0;



    // --------------------------------------------------------------------
    // Flocking parameters (tweak these to change behavior!)
    // 群れのパラメータ（これをいじって挙動を変える！）
    // --------------------------------------------------------------------
    float perceptionRadius = 2.0f;   // view distance / 視野距離
    float separationWeight = 1.0f;   // avoid crowding / 混雑回避
    float alignmentWeight = 1.0f;   // match heading / 向き合わせ
    float cohesionWeight = 1.0f;   // stay together / 集まる
    float maxSpeed = 30.0f;  // speed cap / 速度上限

    float boxSize = 50.0f;

    // --------------------------------------------------------------------
    // Obstacles placed in the world  /  世界に置かれた障害物
    // --------------------------------------------------------------------
    Obstacle obstacles[NUM_OBSTACLES];

    // --------------------------------------------------------------------
    // Scene lifecycle (called by Scene_Manager)
    // シーンのライフサイクル（Scene_Managerが呼ぶ）
    // --------------------------------------------------------------------
    void Start()  override;  // once at scene entry / シーン開始時に1回
    void Update() override;  // every frame logic / 毎フレームのロジック
    void Render() override;  // every frame draw  / 毎フレームの描画
    void End()    override;  // once at scene exit / シーン終了時に1回

    // --------------------------------------------------------------------
    // Setup helpers  /  セットアップ用ヘルパー
    // --------------------------------------------------------------------
    void CameraSetup();           // camera settings / カメラ設定
    void AssetsSetup();           // load model, texture, shader / アセット読み込み
    //void RandomPosRotGenerator(); // random starting positions / 初期位置をランダム生成

    void DrawInstancedFromVBO(Mesh mesh, int count);
};
