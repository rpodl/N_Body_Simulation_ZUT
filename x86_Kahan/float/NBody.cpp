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
    float mass;
    Vector position;
    Vector velocity;
};

float randfloat(std::mt19937_64 &gen, float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen);
}

Body randBody(std::mt19937_64 &gen) {
    float mass = randfloat(gen, 0.5, 5.0);
    Vector pos(randfloat(gen, -1.0, 1.0),
               randfloat(gen, -1.0, 1.0),
               randfloat(gen, -1.0, 1.0));
    Vector vel(randfloat(gen, -0.05, 0.05),
               randfloat(gen, -0.05, 0.05),
               randfloat(gen, -0.05, 0.05));

    return { mass, pos, vel };
}

NBody::NBody(int steps, float orbitRadius) : gc(1.0), bodies(2), timeSteps(steps) {
    masses.resize(2);
    positions.resize(2);
    velocities.resize(2);
    accelerations.resize(2, ORIGIN);

    const float M = 1.0;
    const float r = orbitRadius;
    const float separation = 2.0 * r;
    const float v = std::sqrt(gc * M / (4.0 * r));

    masses[0] = M;
    positions[0]  = Vector( r, 0.0, 0.0);
    velocities[0] = Vector(0.0,  v, 0.0);

    masses[1] = M;
    positions[1]  = Vector(-r, 0.0, 0.0);
    velocities[1] = Vector(0.0, -v, 0.0);

    std::cout << "=== Keplerian two-body test ===\n" << "G = " << gc << " | M = " << M << " | orbit radius r = " << r << " | separation = " << separation << "\n" << "Circular speed v = " << v << "\n" << "Orbital period T = " << (2.0 * M_PI * r / v) << "  (in sim time-units)\n"<< "Steps = " << timeSteps << "\n\n";
}

std::vector<Body> generateRandomBodies(int n, unsigned long long seed) {
    std::mt19937_64 gen(seed);
    std::vector<Body> bodies;

    for (int i = 0; i < n; ++i)
        bodies.push_back(randBody(gen));

    Vector totalVel = ORIGIN;
    Vector totalPos = ORIGIN;
    float totalMass = 0.0;

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

    float maxR = 0.0;
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


    void NBody::computeAccelerations() { //wersja oryginalna
        #pragma omp parallel for
        for (int i = 0; i < bodies; ++i) {
            accelerations[i] = ORIGIN;
	    #pragma omp simd
            for (int j = 0; j < bodies; ++j) {
                if (i != j) {
                    float temp = gc * masses[j] / pow((positions[i] - positions[j]).mod(), 3);
                    accelerations[i] = accelerations[i] + (positions[j] - positions[i]) * temp;
                }
            }
        }
    }


/*
   void NBody::computeAccelerations() { // wersja Kahan
        #pragma omp parallel for
        for (int i = 0; i < bodies; ++i) {
            accelerations[i] = ORIGIN;

            double xs=0; double xe=0;
            double ys=0; double ye=0;
            double zs=0; double ze=0;
         
	    //#pragma omp simd
            for (int j = 0; j < bodies; ++j) {
                if (i != j) {
                    float temp = gc * masses[j] / pow((positions[i] - positions[j]).mod(), 3);
                   
                    float ax = (positions[j].px - positions[i].px) * temp;
                    float ay = (positions[j].py - positions[i].py) * temp;
                    float az = (positions[j].pz - positions[i].pz) * temp;
                   
                //    accelerations[i] = accelerations[i] + (positions[j] - positions[i]) * temp;

                    double xtemp, ytemp, ztemp;
                    double xy, yy, zy;

                    xtemp=xs; xy=ax+xe; xs=xtemp+xy; xe=(xtemp-xs)+xy;
                    ytemp=ys; yy=ay+ye; ys=ytemp+yy; ye=(ytemp-ys)+yy;
                    ztemp=zs; zy=az+ze; zs=ztemp+zy; ze=(ztemp-zs)+zy;
                    

                }

                accelerations[i].px=(float)xs;
                accelerations[i].py=(float)ys;
                accelerations[i].pz=(float)zs;

            }
        }
    }*/



/*  void NBody::computeAccelerations() { // wersja Gill-Moller
        #pragma omp parallel for
        for (int i = 0; i < bodies; ++i) {
            accelerations[i] = ORIGIN;

            float xs=0; float xp=0; float xsold=0;
            float ys=0; float yp=0; float ysold=0;
            float zs=0; float zp=0; float zsold=0;
         
	    #pragma omp simd
            for (int j = 0; j < bodies; ++j) {
                if (i != j) {
                    float temp = gc * masses[j] / pow((positions[i] - positions[j]).mod(), 3);
                   
                    float ax = (positions[j].px - positions[i].px) * temp;
                    float ay = (positions[j].py - positions[i].py) * temp;
                    float az = (positions[j].pz - positions[i].pz) * temp;
                   
                //    accelerations[i] = accelerations[i] + (positions[j] - positions[i]) * temp;

                xs=xsold+ax; xp=xp+(ax-(xs-xsold)); xsold=xs;                    
                ys=ysold+ay; yp=yp+(ay-(ys-ysold)); ysold=ys;                    
                zs=zsold+az; zp=zp+(az-(zs-zsold)); zsold=zs;                    

                }

                accelerations[i].px=xs+xp;
                accelerations[i].py=ys+yp;
                accelerations[i].pz=zs+zp;

            }
        }
    }*/


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


    NBody::NBody(unsigned long long seed, int nBodies, int steps, float gravConst)
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
        const std::chrono::duration<float>es{end-start};
	    std::cout <<"Czas Procesor\t" << std::setprecision(10) << es.count() <<std::endl;
    }
    
    void NBody::simulateKeplerTest() {
    const float M  = masses[0];
    const float r  = positions[0].mod();
    const float v0 = velocities[0].mod();
    const float omega = v0 / r;
    
    std::cout << std::left
              << std::setw(8)  << "Step"
              << std::setw(22) << "Pos error (body 0)"
              << "\n";
    std::cout << std::string(74, '-') << "\n";

    const auto wallStart = std::chrono::steady_clock::now();

    for (int step = 1; step <= timeSteps; ++step) {
        computeAccelerations();
        computePositions();
        computeVelocities();

        const float theta = omega * step;
        Vector exact0(r * std::cos(theta), r * std::sin(theta), 0.0);

        const float posErr = (positions[0] - exact0).mod();


        if (step % std::max(1, timeSteps / 200) == 0 || step == 1) {
            std::cout << std::left
                      << std::setw(8)  << step
                      << std::setw(22) << std::setprecision(6) << posErr
                      << "\n";
        }
    }

    const auto wallEnd = std::chrono::steady_clock::now();
    const std::chrono::duration<float> elapsed{wallEnd - wallStart};

    std::cout << "\nSimulation time: " << std::setprecision(10)
              << elapsed.count() << " s\n";
}

