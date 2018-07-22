#include "Program.h"
#include "../Raytracer/CudaHelp.h"
using namespace TestProgram;

int main(int argc, char** argv)
{

    Program p;
    p.initialize( "Raytracer", 600, 600 );//, 300 );//, 800 );
    p.run();

    return 0;
}