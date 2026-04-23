#pragma once

class Scene_Base { // Abstract base class for all scenes
	public:
		virtual void Start() = 0;
		virtual void Update() = 0;
		virtual void Render() = 0;
		virtual void End() = 0;
};