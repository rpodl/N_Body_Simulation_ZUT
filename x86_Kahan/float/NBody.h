#include <vector>
#include <string>
#include "Vector.h"

class NBody {
private:
    float gc;
    int bodies;
    int timeSteps;
    std::vector<float> masses;
    std::vector<Vector> positions;
    std::vector<Vector> velocities;
    std::vector<Vector> accelerations;

    void resolveCollisions();
    void computeAccelerations();
    void computeVelocities();
    void computePositions();
    
    public:
    
    NBody(unsigned long long seed, int nBodies, int steps, float gravConst = 0.01);
    NBody(int steps, float orbitRadius = 1.0);
    int getTimeSteps();
    void simulate();
    void simulateGPU();
    void simulateBarnesHut();
    void simulateKeplerTest();
    
    friend std::ostream& operator<<(std::ostream&, NBody&);
};
