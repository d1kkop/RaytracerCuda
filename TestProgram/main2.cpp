#include "Program.h"
using namespace TestProgram;

int main(int argc, char** argv)
{

    Program p;
    p.initialize( "Raytracer", 500, 500 );//, 300 );//, 800 );
    p.run();

    return 0;
}