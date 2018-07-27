#include "Program.h"
using namespace TestProgram;

int main(int argc, char** argv)
{

    Program p;
    p.initialize( "Raytracer", 1920, 1080 );//, 300 );//, 800 );
    p.run();

    return 0;
}