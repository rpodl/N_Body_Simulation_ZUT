#include <iostream>
#include <cstdlib>
#include "NBody.h"

int main( int argc, char *argv[ ] ){
    int N = atoi(argv[1]);
    int steps = atoi(argv[2]);
    NBody simulation(44, N, steps, 0.01);
    simulation.simulate();
    //simulation.simulateGPU();
    //simulation.simulateBarnesHut();
    return(0);
}
