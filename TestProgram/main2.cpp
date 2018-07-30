#include "Program.h"
using namespace TestProgram;

int main(int argc, char** argv)
{

    Program p;
    p.initialize( "Raytracer", 500, 500 );
    p.run();

    return 0;
}