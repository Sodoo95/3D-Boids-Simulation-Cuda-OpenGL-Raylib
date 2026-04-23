#include <raylib.h>			// Raylib library for window management, rendering, and input handling
#include "Game.h"			// Include the Game class definition
#include "Globals.h"		// Include global variables and constants
#include "Scene_Manager.h"	// Include the Scene Manager for managing different game scenes
#include "Scene_Menu.h" 	// Include the Menu Scene for the initial game menu

void Game::Start() { // Initialize game resources and states

	// Start the game by setting the initial scene to the menu
	Scene_Manager::GetInstance().changeScene(std::make_unique<Scene_Menu>());
}

void Game::Update() { // Update game logic and states

	// Update the current scene's logic
	Scene_Manager::GetInstance().Update();
}

void Game::Render() { // Render the game visuals

	// Render the current scene's visuals
	Scene_Manager::GetInstance().Render();
}

Game::Game() { // Constructor to initialize the game

	// Initialize the game window with specified width, height, and title
	InitWindow(windowWidth, windowHeight, "Boids sim 3D");	

	// Set the target frames per second to 0 (unlimited)
	SetTargetFPS(0);
}

void Game::Run() { // Main game loop
	Start();

	while (isGameRunning && !WindowShouldClose())
	{
		Update();
		BeginDrawing();
		ClearBackground({ 0, 119, 255, 255 });
		Render();
		EndDrawing();
	}
}


Game::~Game() { // Destructor to clean up resources

	// End the current scene and clean up resources
	Scene_Manager::GetInstance().currentScene->End();
	CloseWindow();
}