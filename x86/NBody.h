#include <vector>
#include <string>
#include "Vector.h"

class NBody {
private:
    double gc;
    int bodies;
    int timeSteps;
    std::vector<double> masses;
    std::vector<Vector> positions;
    std::vector<Vector> velocities;
    std::vector<Vector> accelerations;

    void resolveCollisions();
    void computeAccelerations();
    void computeVelocities();
    void computePositions();
    
    public:
    
    NBody(unsigned long long seed, int nBodies, int steps, double gravConst = 0.01);
    int getTimeSteps();
    void simulate();
    void simulateGPU();
    void simulateBarnesHut();
    
    friend std::ostream& operator<<(std::ostream&, NBody&);
};
