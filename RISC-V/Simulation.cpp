#include <iostream>
#include <cstdlib>
#include <string>
#include "NBody.h"

int main(int argc, char *argv[]) {

    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " N steps [C|G|GB]\n";
        return 1;
    }

    int N = atoi(argv[1]);
    int steps = atoi(argv[2]);
    std::string mode = argv[3];

    NBody simulation(44, N, steps, 0.01);

    if (mode == "C") {
        simulation.simulate();
    }
    else if (mode == "G") {
        //simulation.simulateGPU();
    }
    else if (mode == "GB") {
        //simulation.simulateBarnesHut();
    }
    else {
        std::cout << "Unknown mode. Use C, G, or GB.\n";
        return 1;
    }

    return 0;
}
