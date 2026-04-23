#include <raylib.h>
#include "Scene_Manager.h"
#include "Scene_Menu.h"
#include "Scene_Play.h"
//#include "Globals.h"

void Scene_Menu::Start()
{
}

void Scene_Menu::Update() {
	// Press Enter to start the game
	if (IsKeyPressed(KEY_ENTER)) {	Scene_Manager::GetInstance().changeScene(std::make_unique<Scene_Play>()); }
}

void Scene_Menu::Render() {
	// Clear the background to a light gray color
	DrawText("Menu", 10, 10, 20, DARKGRAY);
}

void Scene_Menu::End()
{
}
