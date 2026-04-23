#pragma once
#include "Scene_Base.h"

// Forward declaration of Scene_Play to avoid circular dependency
class Scene_Menu : public Scene_Base
{
	// Inherited via Scene_Base
	void Start() override;
	void Update() override;
	void Render() override;
	void End() override;
};

