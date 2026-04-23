
// ============================================================================
//  Scene_Play.cpp - main simulation scene implementation
//  Scene_Play.cpp - メインシミュレーションシーンの実装
// ============================================================================
//
//  FRAME FLOW / フレームの流れ:
//    Update():
//      1. poll input (camera, scene switch)  /  入力処理
//      2. BoidsUpdate() runs CUDA kernels    /  CUDAカーネル実行
//      3. build transform matrix per boid on CPU
//         CPU側でボイド毎の変換行列を構築
//    Render():
//      1. draw obstacles (spheres)           /  障害物（球）を描画
//      2. DrawMeshInstanced for all fish     /  全魚をインスタンス描画
//
// ============================================================================

#include <raylib.h>
#include <raymath.h>
#include <rlgl.h>

#include "Scene_Play.h"

// RLIGHTS_IMPLEMENTATION must be defined in EXACTLY ONE .cpp file
// RLIGHTS_IMPLEMENTATIONは必ず1つの.cppファイルだけで定義する
#define RLIGHTS_IMPLEMENTATION
#include "rlights.h"

#include "Scene_Manager.h"
#include "Scene_Menu.h"
#include "CudaCompute.h"
#include <string>


// ============================================================================
//  Start - called once when the scene begins
//  Start - シーン開始時に1回呼ばれる
// ============================================================================
void Scene_Play::Start() {
    CameraSetup();
    AssetsSetup();

    // seed GPU sim with random initial state (use a temp CPU buffer just once)
    Vector4* tmpPos = new Vector4[MAX_BOIDS];
    Vector4* tmpRot = new Vector4[MAX_BOIDS];
    for (int i = 0; i < MAX_BOIDS; i++) {
        tmpPos[i] = { (float)GetRandomValue(-boxSize,boxSize),
                      (float)GetRandomValue(-boxSize,boxSize),
                      (float)GetRandomValue(-boxSize,boxSize), 0 };
        tmpRot[i] = { 1, 0, 0, 0 };  // forward vector, will be overwritten
    }
    BoidsInit((float*)tmpPos, (float*)tmpRot, MAX_BOIDS);
    delete[] tmpPos; delete[] tmpRot;

    SpatialGridInit(MAX_BOIDS, 2.0f, -boxSize, boxSize);
    FlockingInit(perceptionRadius, separationWeight,alignmentWeight, cohesionWeight, maxSpeed);

    for (int i = 0; i < NUM_OBSTACLES; i++)
    {
        obstacles[i] = {
            {(float)GetRandomValue(-boxSize,boxSize),
            (float)GetRandomValue(-boxSize,boxSize),
            (float)GetRandomValue(-boxSize,boxSize)},
        (float)GetRandomValue( 5,15) };
    }
    
    float positions[NUM_OBSTACLES * 3]; float radii[NUM_OBSTACLES];
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        positions[i * 3 + 0] = obstacles[i].pos.x;
        positions[i * 3 + 1] = obstacles[i].pos.y;
        positions[i * 3 + 2] = obstacles[i].pos.z;
        radii[i] = obstacles[i].radius;
    }

    ObstaclesInit(positions, radii, NUM_OBSTACLES);

    // --- interop setup ---
    vboClose = rlLoadVertexBuffer(nullptr, MAX_BOIDS * sizeof(float) * 16, true);
    vboMed = rlLoadVertexBuffer(nullptr, MAX_BOIDS * sizeof(float) * 16, true);
    vboFar = rlLoadVertexBuffer(nullptr, MAX_BOIDS * sizeof(float) * 16, true);

    BoidsRegisterGLBuffers(vboClose, vboMed, vboFar, MAX_BOIDS);

    // attach as per-instance mat4 to each mesh's VAO
    auto attach = [&](Mesh mesh, unsigned int vbo) {
        int loc = instanceMaterial.shader.locs[SHADER_LOC_MATRIX_MODEL];
        rlEnableVertexArray(mesh.vaoId);
        rlEnableVertexBuffer(vbo);
        for (int i = 0; i < 4; i++) {
            rlEnableVertexAttribute(loc + i);
            rlSetVertexAttribute(loc + i, 4, RL_FLOAT, false,
                sizeof(float) * 16,
                sizeof(float) * 4 * i);
            rlSetVertexAttributeDivisor(loc + i, 1);
        }
        rlDisableVertexArray();
        };
    attach(fishModelClose.meshes[0], vboClose);
    attach(fishModelMed.meshes[0], vboMed);
    attach(fishModelFar.meshes[0], vboFar);
}


// ============================================================================
//  Update - called every frame before Render
//  Update - 毎フレーム、Renderの前に呼ばれる
// ============================================================================
void Scene_Play::Update() {
    //UpdateCamera(&camera, CAMERA_PERSPECTIVE);
    if (IsKeyPressed(KEY_ONE)) {
        cameraType = 0;
        camera.position = { 180.0f, 0.0f, 0.0f };
    }
    if (IsKeyPressed(KEY_TWO)) {
        cameraType = 1;
        camera.position = { 10.0f, 0.0f, 0.0f };
    } 
    if (IsKeyPressed(KEY_THREE)) {
        cameraType = 2;
        camera.position = { 180.0f, 0.0f, 0.0f };
    } 

    UpdateCamera(&camera, cameraType);

    if (IsKeyPressed(KEY_TAB)) {
        Scene_Manager::GetInstance().changeScene(std::make_unique<Scene_Menu>());
    }
    float dt = GetFrameTime();

    float sortMs = 0, flockMs = 0;
    BoidsTimedSteps(MAX_BOIDS, dt, &sortMs, &flockMs);

    BoidsBuildMatricesGPU(camera.position.x, camera.position.y, camera.position.z,
        MAX_BOIDS, &countClose, &countMed, &countFar);
    // For debuggin purpose, if needed uncomment, its for console. So dont forget to change int main/WinMain and SubSystem Console/Windows
    //TraceLog(LOG_INFO, "Sort: %.2f  Flock: %.2f", sortMs, flockMs);
}


// ============================================================================
//  Render - called every frame after Update
//  Render - 毎フレーム、Updateの後に呼ばれる
// ============================================================================
void Scene_Play::Render()
{
    BeginMode3D(camera);
    #pragma region GRID WALL CREATION
    rlPushMatrix(); 
    rlTranslatef(0.0f, 0.0f, boxSize);   
    rlRotatef(90, 1, 0, 0);  
    DrawGrid(boxSize, 2.0f);
    rlPopMatrix();

    rlPushMatrix();
    rlTranslatef(0.0f, 0.0f, -boxSize);
    rlRotatef(90, 1, 0, 0);  
    DrawGrid(boxSize, 2.0f);
    rlPopMatrix();

    rlPushMatrix();
    rlTranslatef(0.0f, boxSize, 0.0f);
    rlRotatef(90, 0, 1, 0);  
    DrawGrid(boxSize, 2.0f);
    rlPopMatrix();

    rlPushMatrix();
    rlTranslatef(0.0f, -boxSize, 0.0f);
    rlRotatef(90, 0, 1, 0);  
    DrawGrid(boxSize, 2.0f);
    rlPopMatrix();

    rlPushMatrix();
    rlTranslatef(-boxSize, 0.0f, 0.0f);
    rlRotatef(90, 0, 0, 1);  
    DrawGrid(boxSize, 2.0f);
    rlPopMatrix();
#pragma endregion

    // Draw obstacles slightly smaller than the avoidance radius
    // (fish already start avoiding before touching the sphere)
    // 障害物を回避半径より少し小さく描画
    // （魚は球に触れる前から避け始めるので）
    for (int i = 0; i < NUM_OBSTACLES; i++) {
        DrawSphere(obstacles[i].pos, obstacles[i].radius - 2, { 85, 95, 90, 255 });
    }


    DrawSphere(pointLightPos, 5, YELLOW);

    DrawInstancedFromVBO(fishModelClose.meshes[0], countClose);
    DrawInstancedFromVBO(fishModelMed.meshes[0], countMed);
    DrawInstancedFromVBO(fishModelFar.meshes[0], countFar);

    EndMode3D();
        

    // performance display / FPS表示
    std::string fps = std::to_string(int(1 / GetFrameTime()));
    DrawText(fps.c_str(), 10, 10, 20, GREEN);
    DrawFPS(10, 30);  

    boidCount = "Boids: " + std::to_string(MAX_BOIDS); // const int MAX_BOIDS to string
    DrawText(boidCount.c_str(), 10, 50, 30, BLACK); // Boid count display 
    DrawText(TextFormat("Close (144p): %d\nMed(42p): %d\nFar(21p): %d", countClose, countMed, countFar), 10, 80, 20, BLACK);
}


// ============================================================================
//  End - called once when scene exits
//  End - シーン終了時に1回呼ばれる
// ============================================================================
void Scene_Play::End() {
    BoidsUnregisterGLBuffers();
    rlUnloadVertexBuffer(vboClose);
    rlUnloadVertexBuffer(vboMed);
    rlUnloadVertexBuffer(vboFar);
    UnloadModel(fishModelClose);
    UnloadModel(fishModelMed);
    UnloadModel(fishModelFar);
    UnloadTexture(fishTexture);
    BoidsShutdown();   
}


// ============================================================================
//  Camera setup  /  カメラ設定
// ============================================================================
void Scene_Play::CameraSetup() {
    camera.position = { 180.0f, 0.0f, 0.0f };
    camera.target = { 0.0f,  0.0f,  0.0f };
    camera.up = { 0.0f,  1.0f,  0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    
    // Lock cursor for free-look camera / 自由視点カメラのためマウスロック
    DisableCursor();
}


// ============================================================================
//  Assets setup - load model, texture, shader, lights
//  アセット読み込み - モデル、テクスチャ、シェーダ、ライト
// ============================================================================
void Scene_Play::AssetsSetup() {
    // Load resources  /  リソース読み込み
    fishModelClose = LoadModel("Assets/Herring_LOD4.obj");
    fishModelMed = LoadModel("Assets/Herring_LOD5.obj");
    fishModelFar = LoadModel("Assets/Herring_LOD6.obj");
    fishTexture = LoadTexture("Assets/T_Herring.png");
    instanceMaterial = LoadMaterialDefault();
    instanceMaterial.maps[MATERIAL_MAP_DIFFUSE].color = WHITE;
    instanceMaterial.maps[MATERIAL_MAP_ALBEDO].color = WHITE;

    // Custom shader - instancing vertex shader + lighting fragment shader
    // カスタムシェーダ - インスタンシング頂点 + ライティングフラグメント
    shader = LoadShader("Assets/lighting_instancing.vs", "Assets/lighting.fs");

    // --- Shader uniform setup / シェーダユニフォーム設定 ---

    // "viewPos" uniform = camera position (for specular lighting)
    // "viewPos" = カメラ位置（スペキュラ計算用）
    shader.locs[SHADER_LOC_VECTOR_VIEW] = GetShaderLocation(shader, "viewPos");

    // Ambient color - low-intensity white so dark-side is still visible
    // 環境光 - 弱い白で影側も見えるように
    int ambientLoc = GetShaderLocation(shader, "ambient");
    float ambient[4] = { 0.1f, 0.1f, 0.1f, 0.5f };
    SetShaderValue(shader, ambientLoc, ambient, SHADER_UNIFORM_VEC4);

    // One point light at pointing at origin
    // 原点を照らす点光源に配置
    CreateLight(LIGHT_POINT, pointLightPos, pointLightPos, WHITE, shader);

    // --- Material setup for instanced rendering ---
    // --- インスタンス描画のためのマテリアル設定 ---
    instanceMaterial.shader = shader;

    // Where per-instance transform matrix is plugged into the vertex shader
    // インスタンス毎の変換行列を頂点シェーダに渡す位置
    instanceMaterial.shader.locs[SHADER_LOC_MATRIX_MODEL]
        = GetShaderLocationAttrib(shader, "instanceTransform");

    // Assign texture to albedo map / テクスチャをアルベドマップに
    instanceMaterial.maps[MATERIAL_MAP_ALBEDO].texture = fishTexture;
}

// ============================================================================
//  DrawInstancedFromVBO - issues one instanced draw using a GPU-written VBO
//  DrawInstancedFromVBO - GPUが書き込んだVBOで1回のインスタンス描画を発行
// ============================================================================
//  Why this function exists instead of DrawMeshInstanced:
//  なぜDrawMeshInstancedではなくこの関数なのか：
//
//  raylib's DrawMeshInstanced expects a CPU-side Matrix array and re-uploads
//  it to a VBO every frame. Our matrices already live in a VBO (written by
//  CUDA via interop), so we bypass raylib and talk to rlgl directly.
//
//  raylibのDrawMeshInstancedはCPU側のMatrix配列を受け取り毎フレームVBOへ
//  アップロードする。こちらの行列は(CUDA相互運用で)既にVBOの中にあるので、
//  raylibを経由せずrlglで直接描画する。
// ============================================================================
void Scene_Play::DrawInstancedFromVBO(Mesh mesh, int count)
{
    // Nothing to draw for this LOD this frame - skip the whole setup.
    // このLODは今フレーム描画するものがない - 以下の処理ごとスキップ。
    if (count == 0) return;

    // Bind the lighting+instancing shader program as the active GL program.
    // ライティング+インスタンシング用シェーダをアクティブなGLプログラムにバインド。
    Shader sh = instanceMaterial.shader;
    rlEnableShader(sh.id);

    // -----------------------------------------------------------------------
    //  Camera matrices
    //  カメラ行列
    // -----------------------------------------------------------------------
    // Pull the current view and projection matrices out of rlgl's stack
    // (set earlier by BeginMode3D). Combine them into MVP so the vertex
    // shader can transform vertices: clipPos = MVP * instanceTransform * localPos
    //
    // 現在のビュー/プロジェクション行列をrlglのスタックから取得
    // (BeginMode3Dが設定済み)。MVPに合成して頂点シェーダに渡す。
    // 頂点シェーダは clipPos = MVP * instanceTransform * localPos で変換。
    Matrix matView = rlGetMatrixModelview();
    Matrix matProj = rlGetMatrixProjection();
    Matrix matMVP = MatrixMultiply(matView, matProj);

    // Only upload each uniform if the shader actually has that location.
    // -1 means "not found in shader" and would be an invalid GL call.
    //
    // シェーダにそのuniform位置がある場合のみアップロード。
    // -1は「シェーダに見つからない」という意味で、投げるとGLエラーになる。
    if (sh.locs[SHADER_LOC_MATRIX_VIEW] != -1) rlSetUniformMatrix(sh.locs[SHADER_LOC_MATRIX_VIEW], matView);
    if (sh.locs[SHADER_LOC_MATRIX_PROJECTION] != -1) rlSetUniformMatrix(sh.locs[SHADER_LOC_MATRIX_PROJECTION], matProj);
    if (sh.locs[SHADER_LOC_MATRIX_MVP] != -1) rlSetUniformMatrix(sh.locs[SHADER_LOC_MATRIX_MVP], matMVP);

    // -----------------------------------------------------------------------
    //  Camera position (for specular lighting in the fragment shader)
    //  カメラ位置 (フラグメントシェーダのスペキュラ計算用)
    // -----------------------------------------------------------------------
    // Specular needs the view direction: normalize(cameraPos - fragPos).
    // Without this the fish would have no shiny highlights.
    //
    // スペキュラは視線方向 normalize(cameraPos - fragPos) を必要とする。
    // これを渡さないと魚にハイライトが入らない。
    float viewPos[3] = { camera.position.x, camera.position.y, camera.position.z };
    if (sh.locs[SHADER_LOC_VECTOR_VIEW] != -1)
        SetShaderValue(sh, sh.locs[SHADER_LOC_VECTOR_VIEW], viewPos, SHADER_UNIFORM_VEC3);

    // -----------------------------------------------------------------------
    //  Diffuse color tint
    //  ディフューズ色のティント
    // -----------------------------------------------------------------------
    // The shader multiplies the texture sample by this color. If it's black
    // (the default on some raylib builds when uninitialized), the final
    // color becomes 0 and every fish renders black. We explicitly upload
    // WHITE so the texture passes through unchanged.
    //
    // shader: finalColor = textureSample * colDiffuse * lighting
    //
    // シェーダはテクスチャサンプルをこの色と乗算する。黒だと
    // (raylibのビルドによっては未初期化で黒になる)最終色が0になり、
    // すべての魚が真っ黒になる。白を明示的にアップロードして
    // テクスチャをそのまま通す。
    //
    // シェーダ: 最終色 = テクスチャ * colDiffuse * ライティング
    //
    // raylib stores color as 0-255 bytes (Color struct). The shader wants
    // 0.0-1.0 floats, so divide each channel by 255.
    // raylibは色を0-255のバイトで持つ (Color構造体)。シェーダは0.0-1.0の
    // floatを期待するので各チャンネルを255で割る。
    float color[4] = {
        instanceMaterial.maps[MATERIAL_MAP_DIFFUSE].color.r / 255.0f,
        instanceMaterial.maps[MATERIAL_MAP_DIFFUSE].color.g / 255.0f,
        instanceMaterial.maps[MATERIAL_MAP_DIFFUSE].color.b / 255.0f,
        instanceMaterial.maps[MATERIAL_MAP_DIFFUSE].color.a / 255.0f,
    };
    if (sh.locs[SHADER_LOC_COLOR_DIFFUSE] != -1)
        SetShaderValue(sh, sh.locs[SHADER_LOC_COLOR_DIFFUSE], color, SHADER_UNIFORM_VEC4);

    // -----------------------------------------------------------------------
    //  Texture binding
    //  テクスチャのバインド
    // -----------------------------------------------------------------------
    // Activate texture slot 0 and bind the fish albedo. The fragment shader
    // samples from "texture0" which maps to this slot by convention.
    //
    // テクスチャスロット0を有効化し魚のアルベドをバインド。
    // フラグメントシェーダは慣例に従い"texture0"からサンプリングする。
    rlActiveTextureSlot(0);
    rlEnableTexture(instanceMaterial.maps[MATERIAL_MAP_ALBEDO].texture.id);

    // Normal matrix: transforms normal vectors into view space for lighting.
    // Since our per-instance transforms are pure rotation+translation (no
    // scaling), identity is correct here.
    //
    // 法線行列：ライティング用に法線を変換する行列。
    // インスタンスごとの変換は回転+平行移動のみ(スケールなし)なので
    // identity (単位行列) で正しい。
    if (sh.locs[SHADER_LOC_MATRIX_NORMAL] != -1)
        rlSetUniformMatrix(sh.locs[SHADER_LOC_MATRIX_NORMAL], MatrixIdentity());

    // -----------------------------------------------------------------------
    //  Issue the instanced draw call
    //  インスタンス描画コールを発行
    // -----------------------------------------------------------------------
    // Bind this LOD mesh's VAO. The VAO remembers which VBO holds vertices,
    // which holds indices, and - critically - the per-instance matrix VBO
    // that was attached in Start() with rlSetVertexAttributeDivisor(..., 1).
    //
    // このLODメッシュのVAOをバインド。VAOは頂点VBO・インデックスVBO、
    // そしてStart()でrlSetVertexAttributeDivisor(..., 1)で紐付けた
    // インスタンス行列VBO の全てを記憶している。
    rlEnableVertexArray(mesh.vaoId);

    // Two flavors of instanced draw:
    //   - indexed mesh: draw triangleCount*3 indices, count instances each
    //   - non-indexed:  draw vertexCount verts directly, count instances
    //
    // GL will run the vertex shader (vertexCount * count) times total and
    // feed a different matrix for each instance thanks to the divisor.
    //
    // インスタンス描画の2種類：
    //   - インデックス付き：triangleCount*3個のインデックスをcountインスタンス分
    //   - インデックスなし：vertexCount個の頂点を直接countインスタンス分
    //
    // GLは合計 (vertexCount * count) 回頂点シェーダを実行し、divisorにより
    // インスタンスごとに異なる行列を渡す。
    if (mesh.indices != nullptr)
        rlDrawVertexArrayElementsInstanced(0, mesh.triangleCount * 3, 0, count);
    else
        rlDrawVertexArrayInstanced(0, mesh.vertexCount, count);

    // Unbind VAO to prevent state leaking into the next draw call.
    // VAOのバインド解除 - 次の描画コールに状態が漏れないように。
    rlDisableVertexArray();

    // Clean up: unbind texture and shader. Not strictly required but keeps
    // GL state clean and prevents surprises in other rendering code.
    // 後片付け：テクスチャとシェーダのバインド解除。必須ではないが
    // GLの状態を綺麗に保ち、他の描画コードで予期せぬ動作を防ぐ。
    rlDisableTexture();
    rlDisableShader();
}