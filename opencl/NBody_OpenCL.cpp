#include "NBody.h"

#include <CL/cl.h>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>

static cl_context context;
static cl_command_queue queue;
static cl_program program;

static cl_kernel kernelAccel;
static cl_kernel kernelVel;
static cl_kernel kernelPos;

static int d_N = 0;
static double d_G = 1;

static cl_mem d_masses = nullptr;
static cl_mem d_positions = nullptr;
static cl_mem d_velocities = nullptr;
static cl_mem d_accelerations = nullptr;

void initOpenCL()
{
    cl_platform_id platform;
    cl_device_id device;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);

    std::ifstream file("nbody_kernels.cl");
    std::stringstream buffer;
    buffer << file.rdbuf();

    std::string srcStr = buffer.str();
    const char* src = srcStr.c_str();
    size_t length = srcStr.size();

    program = clCreateProgramWithSource(context, 1, &src, &length, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    kernelAccel = clCreateKernel(program, "computeAccelerationskernel", NULL);
    kernelVel   = clCreateKernel(program, "computeVelocitieskernel", NULL);
    kernelPos   = clCreateKernel(program, "computePositionskernel", NULL);
}

void SendData(int N, const std::vector<double>& masses, const std::vector<Vector>& positions, const std::vector<Vector>& velocities, const std::vector<Vector>& accelerations, double G) {
    d_N = N;
    d_G = G;
    d_masses = clCreateBuffer(context, CL_MEM_READ_ONLY,
                              sizeof(double)*N, NULL, NULL);
    d_positions = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(cl_double3)*N, NULL, NULL);
    d_velocities = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                  sizeof(cl_double3)*N, NULL, NULL);
    d_accelerations = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     sizeof(cl_double3)*N, NULL, NULL);
    std::vector<cl_double3> temp_pos(N), temp_vel(N), temp_acc(N);
    for(size_t i=0;i<N;i++)
    {
        temp_pos[i] = {positions[i].px, positions[i].py, positions[i].pz};
        temp_vel[i] = {velocities[i].px, velocities[i].py, velocities[i].pz};
        temp_acc[i] = {accelerations[i].px, accelerations[i].py, accelerations[i].pz};
    }
    clEnqueueWriteBuffer(queue, d_masses, CL_TRUE, 0,
                         sizeof(double)*N, masses.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_positions, CL_TRUE, 0,
                         sizeof(cl_double3)*N, temp_pos.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_velocities, CL_TRUE, 0,
                         sizeof(cl_double3)*N, temp_vel.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_accelerations, CL_TRUE, 0,
                         sizeof(cl_double3)*N, temp_acc.data(), 0, NULL, NULL);
}


void NBody::simulateGPU(){

    initOpenCL();
    SendData(bodies, masses, positions, velocities, accelerations, gc);
    int N = bodies;
    size_t localSize = 256;
    size_t globalSize = ((N + localSize - 1) / localSize) * localSize;
    const auto start{std::chrono::steady_clock::now()};
    for(size_t ind = 0; ind < getTimeSteps(); ind++){
        clSetKernelArg(kernelAccel,0,sizeof(int),&N);
        clSetKernelArg(kernelAccel,1,sizeof(cl_mem),&d_masses);
        clSetKernelArg(kernelAccel,2,sizeof(cl_mem),&d_positions);
        clSetKernelArg(kernelAccel,3,sizeof(cl_mem),&d_accelerations);
        clSetKernelArg(kernelAccel,4,sizeof(double),&gc);

        clEnqueueNDRangeKernel(queue,kernelAccel,1,NULL,&globalSize,&localSize,0,NULL,NULL);

        clSetKernelArg(kernelPos,0,sizeof(int),&N);
        clSetKernelArg(kernelPos,1,sizeof(cl_mem),&d_positions);
        clSetKernelArg(kernelPos,2,sizeof(cl_mem),&d_velocities);
        clSetKernelArg(kernelPos,3,sizeof(cl_mem),&d_accelerations);

        clEnqueueNDRangeKernel(queue,kernelPos,1,NULL,&globalSize,&localSize,0,NULL,NULL);

        clSetKernelArg(kernelVel,0,sizeof(int),&N);
        clSetKernelArg(kernelVel,1,sizeof(cl_mem),&d_velocities);
        clSetKernelArg(kernelVel,2,sizeof(cl_mem),&d_accelerations);

        clEnqueueNDRangeKernel(queue,kernelVel,1,NULL,&globalSize,&localSize,0,NULL,NULL);
    }
    clFinish(queue);
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double>es{end-start};
	std::cout <<"Czas GPU\t" << std::setprecision(10) << es.count() <<std::endl;
    
    clReleaseMemObject(d_masses);
    clReleaseMemObject(d_positions);
    clReleaseMemObject(d_velocities);
    clReleaseMemObject(d_accelerations);
}
