#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include "NBody.h"

#define EMPTY  -1
#define LOCKED -2

static int d_N = 0;
static double d_G = 1;

static double* d_masses = nullptr;
static double3* d_positions = nullptr;
static double3* d_velocities = nullptr;
static double3* d_accelerations = nullptr;

__host__ __device__ inline double mod(double3 vec)  {
    return sqrtf(vec.x* vec.x + vec.y * vec.y + vec.z * vec.z);
}
    
__host__ __device__ inline double3 operator+(const double3& a, const double3& b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline double3& operator+=(double3& a, const double3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}
__host__ __device__ inline double3 operator-(const double3& a, const double3& b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline double3 operator*(const double3& a, double s) {
    return make_double3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline double3 make_zero() {
    return make_double3(0.0, 0.0, 0.0);
}

// Octree structure inspired by the Rust implementation from:
// https://github.com/DeadlockCode/barnes-hut
struct Oct {
    double3 center;
    double size;
};

struct Node {
    int children;
    int next;
    double3 pos;
    double  mass;
    Oct     oct;
};

struct Octree {
    double t_sq;
    double e_sq;
    std::vector<Node> nodes;
    std::vector<int> parents;

    static const int ROOT = 0;

    Octree(double theta, double eps) {
        t_sq = theta * theta;
        e_sq = eps * eps;
    }

    void clear(const Oct& root_oct) {
        nodes.clear();
        parents.clear();

        Node root;
        root.children = 0;
        root.next = 0;
        root.pos = make_zero();
        root.mass = 0.0;
        root.oct = root_oct;
        nodes.push_back(root);
    }

    inline int find_octant(const Oct& o, const double3& p) const {
        int ox = (p.x > o.center.x) ? 1 : 0;
        int oy = (p.y > o.center.y) ? 1 : 0;
        int oz = (p.z > o.center.z) ? 1 : 0;
        return (oz << 2) | (oy << 1) | ox;
    }

    int subdivide(int node_idx) {
        parents.push_back(node_idx);
        int first = (int)nodes.size();
        nodes[node_idx].children = first;

        int nexts[8] = {
            first + 1, first + 2, first + 3, first + 4,
            first + 5, first + 6, first + 7, nodes[node_idx].next
        };

        double newSize = nodes[node_idx].oct.size * 0.5;

        for (int i = 0; i < 8; ++i) {
            Node c;
            c.children = 0;
            c.next = nexts[i];
            c.pos = make_zero();
            c.mass = 0.0;
            c.oct.size = newSize;

            double sx = ((i & 1) ? 1.0 : 0.0) - 0.5;
            double sy = ((i & 2) ? 1.0 : 0.0) - 0.5;
            double sz = ((i & 4) ? 1.0 : 0.0) - 0.5;

            c.oct.center.x = nodes[node_idx].oct.center.x + sx * newSize;
            c.oct.center.y = nodes[node_idx].oct.center.y + sy * newSize;
            c.oct.center.z = nodes[node_idx].oct.center.z + sz * newSize;

            nodes.push_back(c);
        }
        return first;
    }

    void insert(const double3& pos, double mass) {
        int node = ROOT;

        while (nodes[node].children != 0) {
            int oct = find_octant(nodes[node].oct, pos);
            node = nodes[node].children + oct;
        }

        if (nodes[node].mass == 0.0) {
            nodes[node].pos = pos;
            nodes[node].mass = mass;
            return;
        }

        double3 oldPos = nodes[node].pos;
        double oldMass = nodes[node].mass;

        if (oldPos.x == pos.x && oldPos.y == pos.y && oldPos.z == pos.z) {
            nodes[node].mass += mass;
            return;
        }

        while (true) {
            int first_child = subdivide(node);
            int o1 = find_octant(nodes[node].oct, oldPos);
            int o2 = find_octant(nodes[node].oct, pos);

            if (o1 == o2) {
                node = first_child + o1;
            } else {
                int n1 = first_child + o1;
                int n2 = first_child + o2;

                nodes[n1].pos = oldPos;
                nodes[n1].mass = oldMass;

                nodes[n2].pos = pos;
                nodes[n2].mass = mass;

                return;
            }
        }
    }

    void propagate() {
        for (int k = (int)parents.size() - 1; k >= 0; --k) {
            int node = parents[k];
            int c = nodes[node].children;

            double3 weighted = make_zero();
            double total = 0.0;

            for (int i = 0; i < 8; ++i) {
                weighted += nodes[c + i].pos * nodes[c + i].mass;
                total += nodes[c + i].mass;
            }

            if (total == 0.0) {
                nodes[node].mass = 0.0;
                nodes[node].pos = nodes[node].oct.center;
            } else {
                nodes[node].mass = total;
                nodes[node].pos = weighted * (1.0 / total);
            }
        }
    }
};



void SendData(int N, const std::vector<double>& masses, const std::vector<Vector>& positions, const std::vector<Vector>& velocities, const std::vector<Vector>& accelerations, double G){
    d_N = N;
    d_G = G;
    
    size_t mSize = sizeof(double) * N;
    size_t vectorSize = sizeof(double3) * N;
    cudaMalloc(&d_masses, mSize);
    cudaMalloc(&d_positions, vectorSize);
    cudaMalloc(&d_velocities, vectorSize);
    cudaMalloc(&d_accelerations, vectorSize);
    
    std::vector<double3> temp_pos(N), temp_vel(N), temp_acc(N);
    
    for(size_t ind =0; ind < N; ind++){
        temp_pos[ind] = make_double3(positions[ind].px, positions[ind].py, positions[ind].pz);
        temp_vel[ind] = make_double3(velocities[ind].px, velocities[ind].py, velocities[ind].pz);
        temp_acc[ind] = make_double3(accelerations[ind].px, accelerations[ind].py, accelerations[ind].pz);
    }
    cudaMemcpy(d_masses, masses.data(), mSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, temp_pos.data(), vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, temp_vel.data(), vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_accelerations, temp_acc.data(), vectorSize, cudaMemcpyHostToDevice);
}

void gpuFree(){
    if(d_masses) {
        cudaFree(d_masses);
        d_masses = nullptr;
    }
    if(d_positions) {
        cudaFree(d_positions);
        d_positions = nullptr;
    }
    if(d_velocities){
        cudaFree(d_velocities);
        d_velocities = nullptr;
    }
    if(d_accelerations){
        cudaFree(d_accelerations);
        d_accelerations = nullptr;
    }
}

__global__ void computeAccelerationBarnesHut(const Node* d_nodes,int num_nodes,const double3* d_positions,double3* d_acc,int nBodies,double theta_sq,double eps_sq, double G)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nBodies){
        return;
    }

    double3 pos = d_positions[idx];
    double3 a = make_double3(0.0, 0.0, 0.0);

    int node = 0;
    while (true) {
        const Node n = d_nodes[node];
        double3 d;
        d.x = n.pos.x - pos.x;
        d.y = n.pos.y - pos.y;
        d.z = n.pos.z - pos.z;
        double distSqr = d.x*d.x + d.y*d.y + d.z*d.z + eps_sq;

        if (n.children == 0 || n.oct.size * n.oct.size < distSqr * theta_sq) {
            double invDist = rsqrt(distSqr);
            double invDist3 = invDist * invDist * invDist;

            double s = G * n.mass * invDist3;

            a.x += d.x * s;
            a.y += d.y * s;
            a.z += d.z * s;
            if (n.next == 0) break;
            node = n.next;
        } else {
            node = n.children;
        }
    }

    d_acc[idx] = a;
}

__global__ void computeAccelerationskernel(int N, const double* masses, const double3* pos, double3* acc, double G){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double3 a = make_double3(0.0, 0.0, 0.0);
    const double EPS = 1e-9;
    double3 pi = pos[i];
    for (int j = 0; j < N; j++)
    {
        if (j == i) continue;
        double3 r;
        r.x = pos[j].x - pi.x;
        r.y = pos[j].y - pi.y;
        r.z = pos[j].z - pi.z;
        double distSqr = r.x*r.x + r.y*r.y + r.z*r.z + EPS;
        double invDist = rsqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;
        double s = G * masses[j] * invDist3;
        a.x += r.x * s;
        a.y += r.y * s;
        a.z += r.z * s;
    }
    acc[i] = a;
}

__global__ void computeVelocitieskernel(int N,  double3* vel, const double3*acc){
    int i = blockIdx.x *blockDim.x + threadIdx.x;
    if (i >= N) return;
    vel[i] = vel[i] + acc[i];
}

__global__ void computePositionskernel(int N, double3* pos, const double3* vel, const double3*acc){
    int i = blockIdx.x *blockDim.x + threadIdx.x;
    if (i >= N) return;
    pos[i] = pos[i] + vel[i] + acc[i] * 0.5;
}

void NBody::simulateGPU(){
    
    SendData(bodies, masses, positions, velocities, accelerations, gc);

    int minGridSize =0;
    int blockSize =0;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeAccelerationskernel, 0, 0);
    int gridSize = (d_N + blockSize -1)/blockSize;
    const auto start{std::chrono::steady_clock::now()};
    for(size_t ind = 0; ind < getTimeSteps(); ind++){
        computeAccelerationskernel<<<gridSize, blockSize>>>(d_N, d_masses, d_positions, d_accelerations, d_G);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
        computePositionskernel<<<gridSize, blockSize>>>(d_N, d_positions, d_velocities, d_accelerations);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
        computeVelocitieskernel<<<gridSize, blockSize>>>(d_N, d_velocities, d_accelerations);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
    }
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double>es{end-start};
	std::cout <<"Czas GPU\t" << std::setprecision(10) << es.count() <<std::endl;
    
    gpuFree();
}

void NBody::simulateBarnesHut()
{
    SendData(bodies, masses, positions, velocities, accelerations, gc);
    
    const double theta = 1.0;
    const double eps = 1e-3;
    Node* d_nodes = nullptr;
    const auto start{std::chrono::steady_clock::now()};
    for(size_t ind = 0; ind < getTimeSteps(); ind++){
    double min_x =  1e300, min_y =  1e300, min_z =  1e300;
    double max_x = -1e300, max_y = -1e300, max_z = -1e300;
    for (int i = 0; i < d_N; ++i) {
        double x = positions[i].px;
        double y = positions[i].py;
        double z = positions[i].pz;
        if (x < min_x) min_x = x;
        if (y < min_y) min_y = y;
        if (z < min_z) min_z = z;
        if (x > max_x) max_x = x;
        if (y > max_y) max_y = y;
        if (z > max_z) max_z = z;
    }

    Oct root_oct;
    root_oct.center.x = 0.5 * (min_x + max_x);
    root_oct.center.y = 0.5 * (min_y + max_y);
    root_oct.center.z = 0.5 * (min_z + max_z);

    double side_x = max_x - min_x;
    double side_y = max_y - min_y;
    double side_z = max_z - min_z;
    double max_side = side_x;
    if (side_y > max_side) max_side = side_y;
    if (side_z > max_side) max_side = side_z;
    if (max_side <= 0.0) max_side = 1.0;
    root_oct.size = max_side;

    Octree tree(theta, eps);
    tree.clear(root_oct);

    for (int i = 0; i < d_N; ++i) {
        double3 p = make_double3(positions[i].px, positions[i].py, positions[i].pz);
        tree.insert(p, masses[i]);
    }
    tree.propagate();

    int num_nodes = (int)tree.nodes.size();
    if (num_nodes > 0) {
        cudaError_t err = cudaMalloc(&d_nodes, sizeof(Node) * num_nodes);
        if (err != cudaSuccess) {
            printf("CUDA malloc nodes failed: %s\n", cudaGetErrorString(err));
            gpuFree();
            return;
        }
        err = cudaMemcpy(d_nodes, tree.nodes.data(), sizeof(Node) * num_nodes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("CUDA memcpy nodes failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_nodes);
            gpuFree();
            return;
        }
    }

    int minGridSize =0;
    int blockSize =0;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeAccelerationBarnesHut, 0, 0);
    int gridSize = (d_N + blockSize -1)/blockSize;

    computeAccelerationBarnesHut<<<gridSize, blockSize>>>(
    d_nodes,
    num_nodes,
    d_positions,
    d_accelerations,
    d_N,
    theta * theta,
    eps * eps,
    d_G
);


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error (BH accel): %s\n", cudaGetErrorString(err));
        if (d_nodes) cudaFree(d_nodes);
        gpuFree();
        return;
    }

    computePositionskernel<<<gridSize, blockSize>>>(d_N, d_positions, d_velocities, d_accelerations);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error (positions): %s\n", cudaGetErrorString(err));
        if (d_nodes) cudaFree(d_nodes);
        gpuFree();
        return;
    }

    computeVelocitieskernel<<<gridSize, blockSize>>>(d_N, d_velocities, d_accelerations);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error (velocities): %s\n", cudaGetErrorString(err));
        if (d_nodes) cudaFree(d_nodes);
        gpuFree();
        return;
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA sync error: %s\n", cudaGetErrorString(err));
        if (d_nodes) cudaFree(d_nodes);
        gpuFree();
        return;
    }
    }
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double>es{end-start};
	std::cout <<"Czas Barnes - Hut\t" << std::setprecision(10) << es.count() <<std::endl;
    if (d_nodes) cudaFree(d_nodes);
    gpuFree();
}





