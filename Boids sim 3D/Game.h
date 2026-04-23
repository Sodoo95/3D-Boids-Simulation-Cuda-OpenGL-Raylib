#pragma once

class Game // Main class responsible for managing the game loop and overall game state
{
private:
	void Start();	// Initialize game resources and states
	void Update();	// Update game logic and states
	void Render();	// Render the game visuals
public:
	Game();			// Constructor to initialize the game
	~Game();		// Destructor to clean up resources
	void Run();		// Main game loop
};

