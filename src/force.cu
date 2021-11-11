/*
 * =======================================================================================
 *
 *   Author:   Jan Eitzinger (je), jan.eitzinger@fau.de
 *   Copyright (c) 2021 RRZE, University Erlangen-Nuremberg
 *
 *   This file is part of MD-Bench.
 *
 *   MD-Bench is free software: you can redistribute it and/or modify it
 *   under the terms of the GNU Lesser General Public License as published
 *   by the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   MD-Bench is distributed in the hope that it will be useful, but WITHOUT ANY
 *   WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 *   PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 *   details.
 *
 *   You should have received a copy of the GNU Lesser General Public License along
 *   with MD-Bench.  If not, see <https://www.gnu.org/licenses/>.
 * =======================================================================================
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {
    #include <likwid-marker.h>

    #include <timing.h>
    #include <neighbor.h>
    #include <parameter.h>
    #include <atom.h>
}

void checkError(const char *msg, cudaError_t err)
{
    if (err != cudaSuccess)
    {
        //print a human readable error message
        printf("[CUDA ERROR %s]: %s\r\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}

// cuda kernel
__global__ void calc_force(
    Atom a,
    MD_FLOAT xtmp, MD_FLOAT ytmp, MD_FLOAT ztmp,
    MD_FLOAT *fix, MD_FLOAT *fiy, MD_FLOAT *fiz,
    MD_FLOAT cutforcesq, MD_FLOAT sigma6, MD_FLOAT epsilon,
    int i, int numneighs, int *neighs) {

    // Calculate idx k from thread information
    const long long k = blockIdx.x * blockDim.x + threadIdx.x;
    if( k >= numneighs ) {
        return;
    }

    Atom *atom = &a;

    const int j = neighs[k];
    MD_FLOAT delx = xtmp - atom_x(j);
    MD_FLOAT dely = ytmp - atom_y(j);
    MD_FLOAT delz = ztmp - atom_z(j);
    MD_FLOAT rsq = delx * delx + dely * dely + delz * delz;

#ifdef EXPLICIT_TYPES
    const int type_i = atom->type[i];
    const int type_j = atom->type[j];
    const int type_ij = type_i * atom->ntypes + type_j;
    const MD_FLOAT cutforcesq = atom->cutforcesq[type_ij];
    const MD_FLOAT sigma6 = atom->sigma6[type_ij];
    const MD_FLOAT epsilon = atom->epsilon[type_ij];
#endif

    if(rsq < cutforcesq) {
        MD_FLOAT sr2 = 1.0 / rsq;
        MD_FLOAT sr6 = sr2 * sr2 * sr2 * sigma6;
        MD_FLOAT force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon;
        fix[k] = delx * force;
        fiy[k] = dely * force;
        fiz[k] = delz * force;
    }
}

extern "C" {

double computeForce(
        Parameter *param,
        Atom *atom,
        Neighbor *neighbor
        )
{
    int Nlocal = atom->Nlocal;
    int* neighs;
    MD_FLOAT* fx = atom->fx;
    MD_FLOAT* fy = atom->fy;
    MD_FLOAT* fz = atom->fz;
#ifndef EXPLICIT_TYPES
    MD_FLOAT cutforcesq = param->cutforce * param->cutforce;
    MD_FLOAT sigma6 = param->sigma6;
    MD_FLOAT epsilon = param->epsilon;
#endif

    for(int i = 0; i < Nlocal; i++) {
        fx[i] = 0.0;
        fy[i] = 0.0;
        fz[i] = 0.0;
    }

    Atom c_atom;
    c_atom.Natoms = atom->Natoms;
    c_atom.Nlocal = atom->Nlocal;
    c_atom.Nghost = atom->Nghost;
    c_atom.Nmax = atom->Nmax;
    c_atom.ntypes = atom->ntypes;

    size_t available, total;
    cudaMemGetInfo(&available, &total);
    printf("Available memory: %ldGB\r\n", available / 1024 / 1024 / 1024);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, available);

    // HINT: Run with cuda-memcheck ./MDBench-NVCC in case of error
    // HINT: Only works for data layout = AOS!!!

    checkError( "Malloc1", cudaMalloc((void**)&(c_atom.x), sizeof(MD_FLOAT) * atom->Nmax * 3) );
    checkError( "Memcpy1", cudaMemcpy((void*)(c_atom.x), atom->x, sizeof(MD_FLOAT) * atom->Nmax * 3, cudaMemcpyHostToDevice) );

    checkError( "Malloc4", cudaMalloc((void**)&(c_atom.type), sizeof(int) * atom->Nmax) );
    checkError( "Memcpy4", cudaMemcpy(c_atom.type, atom->type, sizeof(int) * atom->Nmax, cudaMemcpyHostToDevice) );

    checkError( "Malloc5", cudaMalloc((void**)&(c_atom.epsilon), sizeof(MD_FLOAT) * atom->ntypes * atom->ntypes) );
    checkError( "Memcpy5", cudaMemcpy(c_atom.epsilon, atom->epsilon, sizeof(MD_FLOAT) * atom->ntypes * atom->ntypes, cudaMemcpyHostToDevice) );

    checkError( "Malloc6", cudaMalloc((void**)&(c_atom.sigma6), sizeof(MD_FLOAT) * atom->ntypes * atom->ntypes) );
    checkError( "Memcpy6", cudaMemcpy(c_atom.sigma6, atom->sigma6, sizeof(MD_FLOAT) * atom->ntypes * atom->ntypes, cudaMemcpyHostToDevice) );

    checkError( "Malloc7", cudaMalloc((void**)&(c_atom.cutforcesq), sizeof(MD_FLOAT) * atom->ntypes * atom->ntypes) );
    checkError( "Memcpy7", cudaMemcpy(c_atom.cutforcesq, atom->cutforcesq, sizeof(MD_FLOAT) * atom->ntypes * atom->ntypes, cudaMemcpyHostToDevice) );

    double S = getTimeStamp();
    LIKWID_MARKER_START("force");

// #pragma omp parallel for
    for(int i = 0; i < Nlocal; i++) {
        neighs = &neighbor->neighbors[i * neighbor->maxneighs];
        int numneighs = neighbor->numneigh[i];
        MD_FLOAT xtmp = atom_x(i);
        MD_FLOAT ytmp = atom_y(i);
        MD_FLOAT ztmp = atom_z(i);

#ifdef EXPLICIT_TYPES
        const int type_i = atom->type[i];
#endif

        int *c_neighs;
        cudaMalloc((void**)&c_neighs, sizeof(int) * numneighs);
        cudaMemcpy(c_neighs, neighs, sizeof(int) * numneighs, cudaMemcpyHostToDevice);

        MD_FLOAT *c_fix, *c_fiy, *c_fiz;
        cudaMalloc((void**)&c_fix, sizeof(MD_FLOAT) * numneighs);
        cudaMalloc((void**)&c_fiy, sizeof(MD_FLOAT) * numneighs);
        cudaMalloc((void**)&c_fiz, sizeof(MD_FLOAT) * numneighs);

        const int num_blocks = 64;
        const int num_threads_per_block = ceil((float)numneighs / (float)num_blocks);
        // printf("numneighs: %d => num-blocks: %d, num_threads_per_block => %d\r\n", numneighs, num_blocks, num_threads_per_block);

        // launch cuda kernel
        calc_force <<< num_blocks, num_threads_per_block >>> (c_atom, xtmp, ytmp, ztmp, c_fix, c_fiy, c_fiz, cutforcesq, sigma6, epsilon, i, numneighs, c_neighs);
        checkError( "PeekAtLastError", cudaPeekAtLastError() );
        checkError( "DeviceSync", cudaDeviceSynchronize() );

        printf("CUDA done!\r\n");

        // sum result
        MD_FLOAT *d_fix = (MD_FLOAT*)malloc(sizeof(MD_FLOAT) * numneighs);
        MD_FLOAT *d_fiy = (MD_FLOAT*)malloc(sizeof(MD_FLOAT) * numneighs);
        MD_FLOAT *d_fiz = (MD_FLOAT*)malloc(sizeof(MD_FLOAT) * numneighs);
        cudaMemcpy((void**)&d_fix, c_fix, sizeof(MD_FLOAT) * numneighs, cudaMemcpyDeviceToHost);
        cudaMemcpy((void**)&d_fiy, c_fiy, sizeof(MD_FLOAT) * numneighs, cudaMemcpyDeviceToHost);
        cudaMemcpy((void**)&d_fiz, c_fiz, sizeof(MD_FLOAT) * numneighs, cudaMemcpyDeviceToHost);

        printf("COPY ALLOC done!\r\n");

        for(int k = 0; k < numneighs; k++) {
            printf("%d\r\n", k);
            fx[i] += d_fix[k];
            fy[i] += d_fiy[k];
            fz[i] += d_fiz[k];
        }

        printf("COPY done!\r\n");
    }

    LIKWID_MARKER_STOP("force");
    double E = getTimeStamp();

    return E-S;
}
}