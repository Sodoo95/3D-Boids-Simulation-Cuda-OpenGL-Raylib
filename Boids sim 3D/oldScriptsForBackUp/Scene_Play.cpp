
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
//void Scene_Play::Start() {
//    CameraSetup();             // configure the camera / カメラ設定
//    AssetsSetup();             // load model/texture/shader / アセット読み込み
//
//    boidPositions[0] = (Vector4*)AllocPinned(MAX_BOIDS * sizeof(Vector4));
//    boidPositions[1] = (Vector4*)AllocPinned(MAX_BOIDS * sizeof(Vector4));
//    boidRotations[0] = (Vector4*)AllocPinned(MAX_BOIDS * sizeof(Vector4));
//    boidRotations[1] = (Vector4*)AllocPinned(MAX_BOIDS * sizeof(Vector4));
//
//    // seed both so frame 0 has something to draw
//    BoidsInit((float*)boidPositions[0], (float*)boidRotations[0], MAX_BOIDS);
//    memcpy(boidPositions[1], boidPositions[0], MAX_BOIDS * sizeof(Vector4));
//    memcpy(boidRotations[1], boidRotations[0], MAX_BOIDS * sizeof(Vector4));
//
//    RandomPosRotGenerator();   // random initial boid state / 初期状態をランダム化
//
//    // ----- Upload to GPU and configure simulation -----
//    // ----- GPUにアップロードしてシミュレーション設定 -----
//    //BoidsInit((float*)boidPositions, (float*)boidRotations, MAX_BOIDS);
//
//    // Grid: cellSize 4, world box [-30, 30] on each axis → 15x15x15 cells
//    // グリッド：セル4、世界範囲[-30,30] → 15x15x15のセル
//    SpatialGridInit(MAX_BOIDS, 2.0f, -30.0f, 30.0f);
//
//    FlockingInit(
//        perceptionRadius,
//        separationWeight,
//        alignmentWeight,
//        cohesionWeight,
//        maxSpeed
//    );
//
//    // ----- Define and upload obstacles -----
//    // ----- 障害物の定義とアップロード -----
//    obstacles[0] = { {   0,   0,   0 },  8.0f };
//    obstacles[1] = { {  20, -20, -20 }, 10.0f };
//    obstacles[2] = { { -20, -20,  20 }, 10.5f };
//    obstacles[3] = { { -20, -20, -20 }, 10.5f };
//    obstacles[4] = { { -20,  20, -20 }, 10.5f };
//    obstacles[5] = { {  20,  20,  20 }, 10.5f };
//
//    // Pack into flat float arrays for the C API
//    // C APIのために平坦なfloat配列に詰める
//    float positions[NUM_OBSTACLES * 3];
//    float radii[NUM_OBSTACLES];
//    for (int i = 0; i < NUM_OBSTACLES; i++) {
//        positions[i * 3 + 0] = obstacles[i].pos.x;
//        positions[i * 3 + 1] = obstacles[i].pos.y;
//        positions[i * 3 + 2] = obstacles[i].pos.z;
//        radii[i] = obstacles[i].radius;
//    }
//    ObstaclesInit(positions, radii, NUM_OBSTACLES);
//}

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
    FlockingInit(perceptionRadius, separationWeight,
        alignmentWeight, cohesionWeight, maxSpeed);

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
//void Scene_Play::Update() {
//    // Free-fly camera from user input / ユーザー入力で自由移動カメラ
//    UpdateCamera(&camera, CAMERA_FREE);
//
//    // Tab key → switch to menu scene / Tabキーでメニューシーンへ
//    if (IsKeyPressed(KEY_TAB)) {
//        Scene_Manager::GetInstance().changeScene(std::make_unique<Scene_Menu>());
//    }
//
//    double t0 = GetTime();
//
//    // Run GPU simulation step / GPUシミュレーションを1ステップ実行
//    float dt = GetFrameTime();
//
//    int next = 1 - bufIdx;  // write next frame into the other buffer
//
//    // kick off next frame's compute — returns immediately
//    BoidsUpdateAsync((float*)boidPositions[next],
//        (float*)boidRotations[next],
//        MAX_BOIDS, dt);
//
//    countClose = countMed = countFar = 0;
//
//    int readBuf = bufIdx;  // whichever buffer you're reading this frame
//
//    for (int i = 0; i < MAX_BOIDS; i++) {
//        Vector3 fwd = { boidRotations[readBuf][i].x,
//                boidRotations[readBuf][i].y,
//                boidRotations[readBuf][i].z };
//        Vector3 right = Vector3Normalize(Vector3CrossProduct({ 0,1,0 }, fwd));
//        Vector3 up = Vector3CrossProduct(fwd, right);
//
//        Matrix m = {
//            right.x, up.x, fwd.x, boidPositions[readBuf][i].x,
//            right.y, up.y, fwd.y, boidPositions[readBuf][i].y,
//            right.z, up.z, fwd.z, boidPositions[readBuf][i].z,
//                  0,    0,     0, 1
//        };
//
//        float dx = boidPositions[readBuf][i].x - camera.position.x;
//        float dy = boidPositions[readBuf][i].y - camera.position.y;
//        float dz = boidPositions[readBuf][i].z - camera.position.z;
//        float distSq = dx * dx + dy * dy + dz * dz;
//
//        if (distSq < 30 * 30) MtransClose[countClose++] = m;
//        else if (distSq < 60 * 60) MtransMed[countMed++] = m;
//        else                     MtransFar[countFar++] = m;
//    }
//
//    // before next frame starts, make sure GPU finished writing "next"
//    BoidsWaitComplete();
//    bufIdx = next;  // swap — draw will use the freshly-filled buffer
//
//    TraceLog(LOG_INFO, "Matrix Draw: %.3f ms", (GetTime() - t0) * 1000);
//}

void Scene_Play::Update() {
    UpdateCamera(&camera, CAMERA_FREE);
    if (IsKeyPressed(KEY_TAB)) {
        Scene_Manager::GetInstance().changeScene(std::make_unique<Scene_Menu>());
    }



    float dt = GetFrameTime();

    float sortMs = 0, flockMs = 0;
    BoidsTimedSteps(MAX_BOIDS, dt, &sortMs, &flockMs);
    //BoidsSync();  // force wait so timing is accurate
  

    BoidsBuildMatricesGPU(camera.position.x, camera.position.y, camera.position.z,
        MAX_BOIDS, &countClose, &countMed, &countFar);
    // this already syncs internally
  

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
    double t1 = GetTime();

    DrawInstancedFromVBO(fishModelClose.meshes[0], countClose);
    DrawInstancedFromVBO(fishModelMed.meshes[0], countMed);
    DrawInstancedFromVBO(fishModelFar.meshes[0], countFar);

    //TraceLog(LOG_INFO, "Instance Draw: %.3f ms", (GetTime() - t1) * 1000);
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
    camera.position = { 90.0f, 0.0f, 0.0f };
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
//  Random initial positions and rotations
//  ランダムな初期位置と回転
// ============================================================================
//void Scene_Play::RandomPosRotGenerator() {
//    for (int i = 0; i < MAX_BOIDS; i++) {
//        boidPositions[0][i].x = (float)GetRandomValue(-20, 20);
//        boidPositions[0][i].y = (float)GetRandomValue(-20, 20);
//        boidPositions[0][i].z = (float)GetRandomValue(-20, 20);
//
//        boidRotations[0][i].x = (float)GetRandomValue(0, 360);
//        boidRotations[0][i].y = (float)GetRandomValue(0, 360);
//        boidRotations[0][i].z = (float)GetRandomValue(0, 360);
//    }
//}

void Scene_Play::DrawInstancedFromVBO(Mesh mesh, int count) {
    if (count == 0) return;

    Shader sh = instanceMaterial.shader;
    rlEnableShader(sh.id);

    // === matrices ===
    Matrix matView = rlGetMatrixModelview();
    Matrix matProj = rlGetMatrixProjection();
    Matrix matMVP = MatrixMultiply(matView, matProj);

    if (sh.locs[SHADER_LOC_MATRIX_VIEW] != -1) rlSetUniformMatrix(sh.locs[SHADER_LOC_MATRIX_VIEW], matView);
    if (sh.locs[SHADER_LOC_MATRIX_PROJECTION] != -1) rlSetUniformMatrix(sh.locs[SHADER_LOC_MATRIX_PROJECTION], matProj);
    if (sh.locs[SHADER_LOC_MATRIX_MVP] != -1) rlSetUniformMatrix(sh.locs[SHADER_LOC_MATRIX_MVP], matMVP);

    // === view position (for lighting) ===
    float viewPos[3] = { camera.position.x, camera.position.y, camera.position.z };
    if (sh.locs[SHADER_LOC_VECTOR_VIEW] != -1)
        SetShaderValue(sh, sh.locs[SHADER_LOC_VECTOR_VIEW], viewPos, SHADER_UNIFORM_VEC3);

    // === material color (diffuse tint) — THIS is likely why they're black ===
    float color[4] = {
        instanceMaterial.maps[MATERIAL_MAP_DIFFUSE].color.r / 255.0f,
        instanceMaterial.maps[MATERIAL_MAP_DIFFUSE].color.g / 255.0f,
        instanceMaterial.maps[MATERIAL_MAP_DIFFUSE].color.b / 255.0f,
        instanceMaterial.maps[MATERIAL_MAP_DIFFUSE].color.a / 255.0f,
    };
    if (sh.locs[SHADER_LOC_COLOR_DIFFUSE] != -1)
        SetShaderValue(sh, sh.locs[SHADER_LOC_COLOR_DIFFUSE], color, SHADER_UNIFORM_VEC4);

    // === texture ===
    rlActiveTextureSlot(0);
    rlEnableTexture(instanceMaterial.maps[MATERIAL_MAP_ALBEDO].texture.id);
    if (sh.locs[SHADER_LOC_MATRIX_NORMAL] != -1)
        rlSetUniformMatrix(sh.locs[SHADER_LOC_MATRIX_NORMAL], MatrixIdentity());

    // === draw ===
    rlEnableVertexArray(mesh.vaoId);
    if (mesh.indices != nullptr)
        rlDrawVertexArrayElementsInstanced(0, mesh.triangleCount * 3, 0, count);
    else
        rlDrawVertexArrayInstanced(0, mesh.vertexCount, count);
    rlDisableVertexArray();

    rlDisableTexture();
    rlDisableShader();
}