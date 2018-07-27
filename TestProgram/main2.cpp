#include "Program.h"
using namespace TestProgram;

int main(int argc, char** argv)
{

    Program p;
    p.initialize( "Raytracer", 1600, 900 );
    p.run();

    return 0;
}