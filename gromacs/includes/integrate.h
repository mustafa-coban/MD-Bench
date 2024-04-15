/*
 * Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of MD-Bench.
 * Use of this source code is governed by a LGPL-3.0
 * license that can be found in the LICENSE file.
 */
#include <stdbool.h>
//---
#include <atom.h>
#include <parameter.h>
#include <util.h>
#include <timers.h>
#include <timing.h>
#include <simd.h>
/*
void cpuInitialIntegrate(Parameter *param, Atom *atom) {
  
    DEBUG_MESSAGE("cpuInitialIntegrate start\n");
    for(int ci = 0; ci < atom->Nclusters_local; ci++) {
        int ci_vec_base = CI_VECTOR_BASE_INDEX(ci);
        MD_FLOAT *ci_x = &atom->cl_x[ci_vec_base];
        MD_FLOAT *ci_v = &atom->cl_v[ci_vec_base];
        MD_FLOAT *ci_f = &atom->cl_f[ci_vec_base];

        for(int cii = 0; cii < atom->iclusters[ci].natoms; cii++) {
            ci_v[CL_X_OFFSET + cii] += param->dtforce * ci_f[CL_X_OFFSET + cii];
            ci_v[CL_Y_OFFSET + cii] += param->dtforce * ci_f[CL_Y_OFFSET + cii];
            ci_v[CL_Z_OFFSET + cii] += param->dtforce * ci_f[CL_Z_OFFSET + cii];
            ci_x[CL_X_OFFSET + cii] += param->dt * ci_v[CL_X_OFFSET + cii];
            ci_x[CL_Y_OFFSET + cii] += param->dt * ci_v[CL_Y_OFFSET + cii];
            ci_x[CL_Z_OFFSET + cii] += param->dt * ci_v[CL_Z_OFFSET + cii];
        }
    }

    DEBUG_MESSAGE("cpuInitialIntegrate end\n");
}

void  cpuFinalIntegrate(Parameter *param, Atom *atom) {

    DEBUG_MESSAGE("cpuFinalIntegrate start\n");
    for(int ci = 0; ci < atom->Nclusters_local; ci++) {
        int ci_vec_base = CI_VECTOR_BASE_INDEX(ci);
        MD_FLOAT *ci_v = &atom->cl_v[ci_vec_base];
        MD_FLOAT *ci_f = &atom->cl_f[ci_vec_base];

        for(int cii = 0; cii < atom->iclusters[ci].natoms; cii++) {
            ci_v[CL_X_OFFSET + cii] += param->dtforce * ci_f[CL_X_OFFSET + cii];
            ci_v[CL_Y_OFFSET + cii] += param->dtforce * ci_f[CL_Y_OFFSET + cii];
            ci_v[CL_Z_OFFSET + cii] += param->dtforce * ci_f[CL_Z_OFFSET + cii];
        }
    }
    DEBUG_MESSAGE("cpuFinalIntegrate end\n");
}
*/

void cpuInitialIntegrate(Parameter *param, Atom *atom) {
  
    DEBUG_MESSAGE("cpuInitialIntegrate start\n");
    for(int ci = 0; ci < atom->Nclusters_local; ci+=2) {
        int ci_vec_base = CI_VECTOR_BASE_INDEX(ci);
        MD_FLOAT *ci_x = &atom->cl_x[ci_vec_base];
        MD_FLOAT *ci_v = &atom->cl_v[ci_vec_base];
        MD_FLOAT *ci_f = &atom->cl_f[ci_vec_base];

        MD_SIMD_FLOAT dtforce = simd_broadcast(param->dtforce);
        MD_SIMD_FLOAT dt = simd_broadcast(param->dt); 
        
        MD_SIMD_FLOAT vx_vector = simd_fma(simd_load(&ci_f[CL_X_OFFSET]), dtforce, simd_load(&ci_v[CL_X_OFFSET]));
        MD_SIMD_FLOAT vy_vector = simd_fma(simd_load(&ci_f[CL_Y_OFFSET]), dtforce, simd_load(&ci_v[CL_Y_OFFSET]));
        MD_SIMD_FLOAT vz_vector = simd_fma(simd_load(&ci_f[CL_Z_OFFSET]), dtforce, simd_load(&ci_v[CL_Z_OFFSET]));
        MD_SIMD_FLOAT x_vector = simd_fma(vx_vector, dt, simd_load(&ci_x[CL_X_OFFSET]));
        MD_SIMD_FLOAT y_vector = simd_fma(vy_vector, dt, simd_load(&ci_x[CL_Y_OFFSET]));
        MD_SIMD_FLOAT z_vector = simd_fma(vz_vector, dt, simd_load(&ci_x[CL_Z_OFFSET]));
        
        simd_store(&ci_v[CL_X_OFFSET], vx_vector);
        simd_store(&ci_v[CL_Y_OFFSET], vy_vector);
        simd_store(&ci_v[CL_Z_OFFSET], vz_vector);
        simd_store(&ci_x[CL_X_OFFSET], x_vector);
        simd_store(&ci_x[CL_Y_OFFSET], y_vector);
        simd_store(&ci_x[CL_Z_OFFSET], z_vector);
    }

    DEBUG_MESSAGE("cpuInitialIntegrate end\n");
}

void  cpuFinalIntegrate(Parameter *param, Atom *atom) {

    DEBUG_MESSAGE("cpuFinalIntegrate start\n");
    for(int ci = 0; ci < atom->Nclusters_local; ci+=2) {
        int ci_vec_base = CI_VECTOR_BASE_INDEX(ci);
        MD_FLOAT *ci_v = &atom->cl_v[ci_vec_base];
        MD_FLOAT *ci_f = &atom->cl_f[ci_vec_base];

        MD_SIMD_FLOAT dtforce = simd_broadcast(param->dtforce);
        MD_SIMD_FLOAT vx_vector = simd_fma(simd_load(&ci_f[CL_X_OFFSET]), dtforce, simd_load(&ci_v[CL_X_OFFSET]));
        MD_SIMD_FLOAT vy_vector = simd_fma(simd_load(&ci_f[CL_Y_OFFSET]), dtforce, simd_load(&ci_v[CL_Y_OFFSET]));
        MD_SIMD_FLOAT vz_vector = simd_fma(simd_load(&ci_f[CL_Z_OFFSET]), dtforce, simd_load(&ci_v[CL_Z_OFFSET]));
        simd_store(&ci_v[CL_X_OFFSET], vx_vector);
        simd_store(&ci_v[CL_Y_OFFSET], vy_vector);
        simd_store(&ci_v[CL_Z_OFFSET], vz_vector);
    }

    DEBUG_MESSAGE("cpuFinalIntegrate end\n");
}

#ifdef CUDA_TARGET
void cudaInitialIntegrate(Parameter*, Atom*);
void cudaFinalIntegrate(Parameter*, Atom*);
#endif


   