/*
 * Copyright (C)  NHR@FAU, University Erlangen-Nuremberg.
 * All rights reserved. This file is part of MD-Bench.
 * Use of this source code is governed by a LGPL-3.0
 * license that can be found in the LICENSE file.
 */
#ifndef __SIMD_H__
#define __SIMD_H__

#ifndef CLUSTER_M
#define CLUSTER_M 1
#endif

#ifndef CLUSTER_N
#define CLUSTER_N 1
#endif

#if (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#ifndef NO_ZMM_INTRIN
#include <zmmintrin.h>
#endif

#if defined(__ISA_AVX512__)
#if PRECISION == 2
#include "simd/avx512_double.h"
#else
#include "simd/avx512_float.h"
#endif
#endif

#if defined(__ISA_AVX2__)
#if PRECISION == 2
#include "simd/avx2_double.h"
#else
#include "simd/avx2_float.h"
#endif
#endif

#if defined(__ISA_AVX__)
#if PRECISION == 2
#include "simd/avx_double.h"
#else
#include "simd/avx_float.h"
#endif
#endif
#endif

#ifdef __ARM_NEON
#include <arm_acle.h>
#include <arm_neon.h>
#endif
#ifdef __ARM_FEATURE_SVE
#include <arm_acle.h>
#include <arm_sve.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIMD_PRINT_REAL(a) simd_print_real(#a, a);
#define SIMD_PRINT_MASK(a) simd_print_mask(#a, a);

extern unsigned int simd_mask_to_u32(MD_SIMD_MASK a);

static inline void simd_print_real(const char* ref, MD_SIMD_FLOAT a)
{
    double x[VECTOR_WIDTH];
    memcpy(x, &a, sizeof(x));

    fprintf(stdout, "%s: ", ref);
    for (int i = 0; i < VECTOR_WIDTH; i++) {
        fprintf(stdout, "%f ", x[i]);
    }

    fprintf(stdout, "\n");
}

static inline void simd_print_mask(const char* ref, MD_SIMD_MASK a)
{
    fprintf(stdout, "%s: %x\n", ref, simd_mask_to_u32(a));
}

#endif // __SIMD_H__
