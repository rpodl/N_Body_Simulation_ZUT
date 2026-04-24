#include <riscv_vector.h>
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

inline double reduce_sum(vfloat64m1_t v, size_t vl) {
    vfloat64m1_t tmp = __riscv_vfredusum_vs_f64m1_f64m1(
        v,
        __riscv_vfmv_v_f_f64m1(0.0, 1),
        vl
    );
    return __riscv_vfmv_f_s_f64m1_f64(tmp);
}

/*
void NBody::computeAccelerations() {
        #pragma omp parallel for
for (int i = 0; i < bodies; ++i) {

    double ax=0, ay=0, az=0;

    #pragma omp simd reduction(+:ax,ay,az)
    for (int j = 0; j < bodies; ++j) {

        if (i == j) continue;

        double dx = positions[j].px - positions[i].px;
        double dy = positions[j].py - positions[i].py;
        double dz = positions[j].pz - positions[i].pz;

        double dist = sqrt(dx*dx + dy*dy + dz*dz);
        double inv = gc * masses[j] / (dist*dist*dist);

        ax += dx * inv;
        ay += dy * inv;
        az += dz * inv;
    }

    accelerations[i] = Vector(ax, ay, az);
}
    }
*/
void NBody::computeAccelerations() {

    int N = bodies;
double* posx = (double*) aligned_alloc(64, bodies * sizeof(double));
double* posy = (double*) aligned_alloc(64, bodies * sizeof(double));
double* posz = (double*) aligned_alloc(64, bodies * sizeof(double));

for (int j = 0; j < bodies; ++j) {
    posx[j] = positions[j].px;
    posy[j] = positions[j].py;
    posz[j] = positions[j].pz;
}
    
    #pragma omp parallel for
for (int i = 0; i < N; i++) {

    double axi = 0.0, ayi = 0.0, azi = 0.0;

    double pix = posx[i];
    double piy = posy[i];
    double piz = posz[i];
    
    

    int j = 0;
    while (j < N) {

        size_t vl = __riscv_vsetvl_e64m1(N - j);

        // load j
        vfloat64m1_t xj = __riscv_vle64_v_f64m1(&posx[j], vl);
        vfloat64m1_t yj = __riscv_vle64_v_f64m1(&posy[j], vl);
        vfloat64m1_t zj = __riscv_vle64_v_f64m1(&posz[j], vl);

        // broadcast i
        vfloat64m1_t xi = __riscv_vfmv_v_f_f64m1(pix, vl);
        vfloat64m1_t yi = __riscv_vfmv_v_f_f64m1(piy, vl);
        vfloat64m1_t zi = __riscv_vfmv_v_f_f64m1(piz, vl);

        // dx
        vfloat64m1_t dx = __riscv_vfsub_vv_f64m1(xj, xi, vl);
        vfloat64m1_t dy = __riscv_vfsub_vv_f64m1(yj, yi, vl);
        vfloat64m1_t dz = __riscv_vfsub_vv_f64m1(zj, zi, vl);

        // dist^2
        vfloat64m1_t r2 = __riscv_vfmul_vv_f64m1(dx, dx, vl);
        r2 = __riscv_vfmacc_vv_f64m1(r2, dy, dy, vl);
        r2 = __riscv_vfmacc_vv_f64m1(r2, dz, dz, vl);

        // epsilon (unikasz i==j)
        vfloat64m1_t eps = __riscv_vfmv_v_f_f64m1(1e-9, vl);
        r2 = __riscv_vfadd_vv_f64m1(r2, eps, vl);

        // inv = 1 / sqrt(r2)^3
        vfloat64m1_t inv_r = __riscv_vfrsqrt7_v_f64m1(r2, vl); // approx
        vfloat64m1_t inv_r2 = __riscv_vfmul_vv_f64m1(inv_r, inv_r, vl);
        vfloat64m1_t inv_r3 = __riscv_vfmul_vv_f64m1(inv_r2, inv_r, vl);

        // masses
        vfloat64m1_t mj = __riscv_vle64_v_f64m1(&masses[j], vl);

        vfloat64m1_t inv = __riscv_vfmul_vv_f64m1(mj, inv_r3, vl);
        inv = __riscv_vfmul_vf_f64m1(inv, gc, vl);

        // acc
        vfloat64m1_t axv = __riscv_vfmul_vv_f64m1(dx, inv, vl);
        vfloat64m1_t ayv = __riscv_vfmul_vv_f64m1(dy, inv, vl);
        vfloat64m1_t azv = __riscv_vfmul_vv_f64m1(dz, inv, vl);

        // redukcja
axi += reduce_sum(axv, vl);
ayi += reduce_sum(ayv, vl);
azi += reduce_sum(azv, vl);

        j += vl;
    }

    accelerations[i] = Vector(axi, ayi, azi);
}
}
/*    void NBody::computeAccelerations() {
        #pragma omp parallel for simd
        for (int i = 0; i < bodies; ++i) {
            accelerations[i] = ORIGIN;
	    //#pragma omp simd
            for (int j = 0; j < bodies; ++j) {
                if (i != j) {
                    double temp = gc * masses[j] / pow((positions[i] - positions[j]).mod(), 3);
                    accelerations[i] = accelerations[i] + (positions[j] - positions[i]) * temp;
                }
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
