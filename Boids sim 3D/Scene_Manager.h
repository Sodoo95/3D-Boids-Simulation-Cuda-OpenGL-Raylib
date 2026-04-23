#pragma once
#include <iostream>
#include "Scene_Base.h"

// Singleton Scene Manager to handle scene transitions and updates
class Scene_Manager {
private:
	Scene_Manager() : currentScene(nullptr) {}				// Private constructor for singleton pattern
	Scene_Manager(const Scene_Manager&) = delete;			// Delete copy constructor
	Scene_Manager& operator=(const Scene_Manager&) = delete;// Delete copy assignment operator
public:
	std::unique_ptr<Scene_Base> currentScene;				// Current active scene

	static Scene_Manager& GetInstance() {					// Static method to access the singleton instance
		static Scene_Manager instance;
		return instance;
	}

	void changeScene(std::unique_ptr<Scene_Base> newScene) {// Change the current scene and start the new one
		currentScene = std::move(newScene);
		currentScene->Start();
	}

	void Update() {											// Update the current scene if it exists
		if (currentScene) {
			currentScene->Update();
		}
	}

	void Render() {											// Render the current scene if it exists
		if (currentScene) {
			currentScene->Render();
		}
	}
	void End() {											// End the current scene if it exists
		if (currentScene) {
			currentScene->End();
		}
	}
};