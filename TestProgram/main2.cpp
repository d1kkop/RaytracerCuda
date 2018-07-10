#include "Program.h"
#include "../Raytracer/CudaHelp.h"
using namespace TestProgram;

int main(int argc, char** argv)
{

    Program p;
    p.initialize( "Raytracer",1200, 800 );
    p.run();

    return 0;
}