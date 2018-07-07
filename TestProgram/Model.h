#pragma once

#include <string>
#include "../Raytracer/Scene.h"
#include "../Raytracer/Types.h"


namespace TestProgram
{
    class Model
    {
    public:
        static bool load(const std::string& name, Beam::sptr<Beam::IScene>& toScene);
    };
}