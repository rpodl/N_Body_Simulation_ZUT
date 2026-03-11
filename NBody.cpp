#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>
#include <omp.h>
#include <chrono>
#include "NBody.h"

const Vector ORIGIN{ 0.0, 0.0, 0.0 };

struct Body {
    double mass;
    Vector position;
    Vector velocity;
};

double randDouble(std::mt19937_64 &gen, double min, double max) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(gen);
}

Body randBody(std::mt19937_64 &gen) {
    double mass = randDouble(gen, 0.5, 5.0);
    Vector pos(randDouble(gen, -1.0, 1.0),
               randDouble(gen, -1.0, 1.0),
               randDouble(gen, -1.0, 1.0));
    Vector vel(randDouble(gen, -0.05, 0.05),
               randDouble(gen, -0.05, 0.05),
               randDouble(gen, -0.05, 0.05));

    return { mass, pos, vel };
}

std::vector<Body> generateRandomBodies(int n, unsigned long long seed) {
    std::mt19937_64 gen(seed);
    std::vector<Body> bodies;

    for (int i = 0; i < n; ++i)
        bodies.push_back(randBody(gen));

    Vector totalVel = ORIGIN;
    Vector totalPos = ORIGIN;
    double totalMass = 0.0;

    for (auto &b : bodies) {
        totalVel = totalVel + (b.velocity * b.mass);
        totalPos = totalPos + (b.position * b.mass);
        totalMass += b.mass;
    }

    Vector avgVel = totalVel * (1.0 / totalMass);
    Vector avgPos = totalPos * (1.0 / totalMass);

    for (auto &b : bodies) {
        b.velocity = b.velocity - avgVel;
        b.position = b.position - avgPos;
    }

    double maxR = 0.0;
    for (auto &b : bodies)
        maxR = std::max(maxR, b.position.mod());
    if (maxR > 0) {
        for (auto &b : bodies)
            b.position = b.position * (1.0 / maxR);
    }

    return bodies;
}

void NBody::resolveCollisions() {
    for (int i = 0; i < bodies; ++i) {
        for (int j = i + 1; j < bodies; ++j) {
            if (positions[i] == positions[j]) {
                std::swap(velocities[i], velocities[j]);
            }
        }
    }
}

    void NBody::computeAccelerations() {
        #pragma omp parallel for
        for (int i = 0; i < bodies; ++i) {
            accelerations[i] = ORIGIN;
	    #pragma omp simd
            for (int j = 0; j < bodies; ++j) {
                if (i != j) {
                    double temp = gc * masses[j] / pow((positions[i] - positions[j]).mod(), 3);
                    accelerations[i] = accelerations[i] + (positions[j] - positions[i]) * temp;
                }
            }
        }
    }

    void NBody::computeVelocities() {
        #pragma omp parallel for
        for (int i = 0; i < bodies; ++i) {
            velocities[i] = velocities[i] + accelerations[i];
        }
    }

    void NBody::computePositions() {
        #pragma omp parallel for
        for (int i = 0; i < bodies; ++i) {
            positions[i] = positions[i] + velocities[i] + accelerations[i] * 0.5;
        }
    }
    
    std::ostream& operator<<(std::ostream& out, NBody& nb) {
    for (int i = 0; i < nb.bodies; ++i) {
        out << "Body " << i + 1 << " : ";
        out << std::setprecision(6) << std::setw(9) << nb.positions[i];
        out << " | ";
        out << std::setprecision(6) << std::setw(9) << nb.velocities[i];
        out << '\n';
    }
    return out;
}


    NBody::NBody(unsigned long long seed, int nBodies, int steps, double gravConst)
    : gc(gravConst), bodies(nBodies), timeSteps(steps)
{
    auto randomBodies = generateRandomBodies(nBodies, seed);

    masses.resize(nBodies);
    positions.resize(nBodies);
    velocities.resize(nBodies);
    accelerations.resize(nBodies, ORIGIN);

    for (int i = 0; i < nBodies; ++i) {
        masses[i] = randomBodies[i].mass;
        positions[i] = randomBodies[i].position;
        velocities[i] = randomBodies[i].velocity;
    }

    std::cout << "Generated " << nBodies << " random bodies with seed " << seed << "\n";
    std::cout << "G = " << gc << ", Steps = " << timeSteps << "\n";
}

    int NBody::getTimeSteps() {
        return timeSteps;
    }

    void NBody::simulate() {
        const auto start{std::chrono::steady_clock::now()};
        for (size_t ind = 0; ind < getTimeSteps(); ++ind) {
        
            //std::cout << "\nCycle " << ind + 1 << '\n';
            computeAccelerations();
            computePositions();
            computeVelocities();
            //resolveCollisions();
            //std::cout << *this;
        }
        const auto end{std::chrono::steady_clock::now()};
        const std::chrono::duration<double>es{end-start};
	    std::cout <<"Czas Procesor\t" << std::setprecision(10) << es.count() <<std::endl;
    }
    
    

