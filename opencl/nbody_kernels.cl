#if __OPENCL_VERSION__ <= CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif


__kernel void computeVelocitieskernel( const int N, __global double3* vel, __global const double3* acc) {
    int i = get_global_id(0);
    if (i >= N) return;

    vel[i] = vel[i] + acc[i];
}

__kernel void computePositionskernel( const int N, __global double3* pos, __global const double3* vel, __global const double3* acc) {
    int i = get_global_id(0);
    if (i >= N) return;

    pos[i] = pos[i] + vel[i] + acc[i] * 0.5;
}

__kernel void computeAccelerationskernel( const int N, __global const double* masses, __global const double3* pos, __global double3* acc, const double G) {
    int i = get_global_id(0);
    if (i >= N) return;

    double3 a = make_zero();
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
