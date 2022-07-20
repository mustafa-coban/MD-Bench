/*
 * =======================================================================================
 *
 *   Author:   Jan Eitzinger (je), jan.eitzinger@fau.de
 *   Copyright (c) 2020 RRZE, University Erlangen-Nuremberg
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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <limits.h>
#include <math.h>
#include <float.h>

#include <likwid-marker.h>

#include <timing.h>
#include <allocate.h>
#include <neighbor.h>
#include <parameter.h>
#include <atom.h>
#include <stats.h>
#include <thermo.h>
#include <pbc.h>
#include <timers.h>
#include <eam.h>
#include <vtk.h>

#define HLINE "----------------------------------------------------------------------------\n"

extern void cuda_final_integrate(bool doReneighbour, Parameter *param,
                                 Atom *atom, Atom *c_atom,
                                 const int num_threads_per_block);
extern void cuda_initial_integrate(bool doReneighbour, Parameter *param,
                                   Atom *atom, Atom *c_atom,
                                   const int num_threads_per_block);

extern double computeForce(bool, Parameter*, Atom*, Neighbor*, Atom*, Neighbor*, const int);
extern double computeForceTracing(Parameter*, Atom*, Neighbor*, Stats*, int, int);
extern double computeForceEam(Eam* eam, Parameter*, Atom *atom, Neighbor *neighbor, Stats *stats, int first_exec, int timestep);

void init(Parameter *param)
{
    param->input_file = NULL;
    param->vtk_file = NULL;
    param->force_field = FF_LJ;
    param->epsilon = 1.0;
    param->sigma6 = 1.0;
    param->rho = 0.8442;
    param->ntypes = 4;
    param->ntimes = 200;
    param->dt = 0.005;
    param->nx = 32;
    param->ny = 32;
    param->nz = 32;
    param->cutforce = 2.5;
    param->cutneigh = param->cutforce + 0.30;
    param->temp = 1.44;
    param->nstat = 100;
    param->mass = 1.0;
    param->dtforce = 0.5 * param->dt;
    param->every = 20;
    param->proc_freq = 2.4;
}

void initCudaAtom(Atom *atom, Neighbor *neighbor, Atom *c_atom, Neighbor *c_neighbor) {

    c_atom->Natoms = atom->Natoms;
    c_atom->Nlocal = atom->Nlocal;
    c_atom->Nghost = atom->Nghost;
    c_atom->Nmax = atom->Nmax;
    c_atom->ntypes = atom->ntypes;

    c_atom->border_map = NULL;

    const int Nlocal = atom->Nlocal;

    checkCUDAError( "c_atom->x malloc", cudaMalloc((void**)&(c_atom->x), sizeof(MD_FLOAT) * atom->Nmax * 3) );
    checkCUDAError( "c_atom->x memcpy", cudaMemcpy(c_atom->x, atom->x, sizeof(MD_FLOAT) * atom->Nmax * 3, cudaMemcpyHostToDevice) );

    checkCUDAError( "c_atom->fx malloc", cudaMalloc((void**)&(c_atom->fx), sizeof(MD_FLOAT) * Nlocal * 3) );

    checkCUDAError( "c_atom->vx malloc", cudaMalloc((void**)&(c_atom->vx), sizeof(MD_FLOAT) * Nlocal * 3) );
    checkCUDAError( "c_atom->vx memcpy", cudaMemcpy(c_atom->vx, atom->vx, sizeof(MD_FLOAT) * Nlocal * 3, cudaMemcpyHostToDevice) );

    checkCUDAError( "c_atom->type malloc", cudaMalloc((void**)&(c_atom->type), sizeof(int) * atom->Nmax) );
    checkCUDAError( "c_atom->epsilon malloc", cudaMalloc((void**)&(c_atom->epsilon), sizeof(MD_FLOAT) * atom->ntypes * atom->ntypes) );
    checkCUDAError( "c_atom->sigma6 malloc", cudaMalloc((void**)&(c_atom->sigma6), sizeof(MD_FLOAT) * atom->ntypes * atom->ntypes) );
    checkCUDAError( "c_atom->cutforcesq malloc", cudaMalloc((void**)&(c_atom->cutforcesq), sizeof(MD_FLOAT) * atom->ntypes * atom->ntypes) );

    checkCUDAError( "c_neighbor->neighbors malloc", cudaMalloc((void**)&c_neighbor->neighbors, sizeof(int) * Nlocal * neighbor->maxneighs) );
    checkCUDAError( "c_neighbor->numneigh malloc", cudaMalloc((void**)&c_neighbor->numneigh, sizeof(int) * Nlocal) );

    checkCUDAError( "c_atom->type memcpy", cudaMemcpy(c_atom->type, atom->type, sizeof(int) * atom->Nmax, cudaMemcpyHostToDevice) );
    checkCUDAError( "c_atom->sigma6 memcpy", cudaMemcpy(c_atom->sigma6, atom->sigma6, sizeof(MD_FLOAT) * atom->ntypes * atom->ntypes, cudaMemcpyHostToDevice) );
    checkCUDAError( "c_atom->epsilon memcpy", cudaMemcpy(c_atom->epsilon, atom->epsilon, sizeof(MD_FLOAT) * atom->ntypes * atom->ntypes, cudaMemcpyHostToDevice) );

    checkCUDAError( "c_atom->cutforcesq memcpy", cudaMemcpy(c_atom->cutforcesq, atom->cutforcesq, sizeof(MD_FLOAT) * atom->ntypes * atom->ntypes, cudaMemcpyHostToDevice) );
}


double setup(
        Parameter *param,
        Eam *eam,
        Atom *atom,
        Neighbor *neighbor,
        Atom *c_atom,
        Neighbor *c_neighbor,
        Stats *stats,
        const int num_threads_per_block,
	double* timers)
{
    if(param->force_field == FF_EAM) { initEam(eam, param); }
    double S, E;
    param->lattice = pow((4.0 / param->rho), (1.0 / 3.0));
    param->xprd = param->nx * param->lattice;
    param->yprd = param->ny * param->lattice;
    param->zprd = param->nz * param->lattice;

    S = getTimeStamp();
    initAtom(atom);
    initNeighbor(neighbor, param);
    initPbc(atom);
    initStats(stats);
    setupNeighbor();
    createAtom(atom, param);
    setupThermo(param, atom->Natoms);
    adjustThermo(param, atom);
    setupPbc(atom, param);
    initCudaAtom(atom, neighbor, c_atom, c_neighbor);
    updatePbc_cuda(atom, param, c_atom, true, num_threads_per_block);
    buildNeighbor_cuda(atom, neighbor, c_atom, c_neighbor, num_threads_per_block, timers);
    E = getTimeStamp();


    return E-S;
}

double reneighbour(
        Parameter *param,
        Atom *atom,
        Neighbor *neighbor,
        Atom *c_atom,
        Neighbor *c_neighbor,
        const int num_threads_per_block,
	double* timers)
{
    double S, E, beforeEvent, afterEvent;

    S = getTimeStamp();
    beforeEvent = S;
    LIKWID_MARKER_START("reneighbour");
    updateAtomsPbc_cuda(atom, param, c_atom, num_threads_per_block);
    afterEvent = getTimeStamp();
    timers[NEIGH_UPDATE_ATOMS_PBC] += afterEvent - beforeEvent;
    beforeEvent = afterEvent;
    setupPbc(atom, param);
    afterEvent = getTimeStamp();
    timers[NEIGH_SETUP_PBC] += afterEvent - beforeEvent;
    beforeEvent = afterEvent;
    updatePbc_cuda(atom, param, c_atom, true, num_threads_per_block);
    afterEvent = getTimeStamp();
    timers[NEIGH_UPDATE_PBC] += afterEvent - beforeEvent;
    beforeEvent = afterEvent;
    //sortAtom(atom);
    buildNeighbor_cuda(atom, neighbor, c_atom, c_neighbor, num_threads_per_block, timers);
    LIKWID_MARKER_STOP("reneighbour");
    E = getTimeStamp();
    afterEvent = E;
    timers[NEIGH_BUILD_LISTS] += afterEvent - beforeEvent;

    return E-S;
}

void initialIntegrate(Parameter *param, Atom *atom)
{
    for(int i = 0; i < atom->Nlocal; i++) {
        atom_vx(i) += param->dtforce * atom_fx(i);
        atom_vy(i) += param->dtforce * atom_fy(i);
        atom_vz(i) += param->dtforce * atom_fz(i);
        atom_x(i) = atom_x(i) + param->dt * atom_vx(i);
        atom_y(i) = atom_y(i) + param->dt * atom_vy(i);
        atom_z(i) = atom_z(i) + param->dt * atom_vz(i);
    }
}

void finalIntegrate(Parameter *param, Atom *atom)
{
    for(int i = 0; i < atom->Nlocal; i++) {
        atom_vx(i) += param->dtforce * atom_fx(i);
        atom_vy(i) += param->dtforce * atom_fy(i);
        atom_vz(i) += param->dtforce * atom_fz(i);
    }
}

void printAtomState(Atom *atom)
{
    printf("Atom counts: Natoms=%d Nlocal=%d Nghost=%d Nmax=%d\n",
            atom->Natoms, atom->Nlocal, atom->Nghost, atom->Nmax);

    /*     int nall = atom->Nlocal + atom->Nghost; */

    /*     for (int i=0; i<nall; i++) { */
    /*         printf("%d  %f %f %f\n", i, atom->x[i], atom->y[i], atom->z[i]); */
    /*     } */
}

int str2ff(const char *string)
{
    if(strncmp(string, "lj", 2) == 0) return FF_LJ;
    if(strncmp(string, "eam", 3) == 0) return FF_EAM;
    return -1;
}

const char* ff2str(int ff)
{
    if(ff == FF_LJ) { return "lj"; }
    if(ff == FF_EAM) { return "eam"; }
    return "invalid";
}

int get_num_threads() {

    const char *num_threads_env = getenv("NUM_THREADS");
    int num_threads = 0;
    if(num_threads_env == 0)
        num_threads = 32;
    else {
        num_threads = atoi(num_threads_env);
    }

    return num_threads;
}

int main(int argc, char** argv)
{
    double timer[NUMTIMER];
    Eam eam;
    Atom atom;
    Neighbor neighbor;
    Stats stats;
    Parameter param;
    Atom c_atom;
    Neighbor c_neighbor;

    LIKWID_MARKER_INIT;
#pragma omp parallel
    {
        LIKWID_MARKER_REGISTER("force");
        //LIKWID_MARKER_REGISTER("reneighbour");
        //LIKWID_MARKER_REGISTER("pbc");
    }
    init(&param);

    for(int i = 0; i < argc; i++)
    {
        if((strcmp(argv[i], "-f") == 0))
        {
            if((param.force_field = str2ff(argv[++i])) < 0) {
                fprintf(stderr, "Invalid force field!\n");
                exit(-1);
            }
            continue;
        }
        if((strcmp(argv[i], "-i") == 0))
        {
            param.input_file = strdup(argv[++i]);
            continue;
        }
        if((strcmp(argv[i], "-n") == 0) || (strcmp(argv[i], "--nsteps") == 0))
        {
            param.ntimes = atoi(argv[++i]);
            continue;
        }
        if((strcmp(argv[i], "-nx") == 0))
        {
            param.nx = atoi(argv[++i]);
            continue;
        }
        if((strcmp(argv[i], "-ny") == 0))
        {
            param.ny = atoi(argv[++i]);
            continue;
        }
        if((strcmp(argv[i], "-nz") == 0))
        {
            param.nz = atoi(argv[++i]);
            continue;
        }
        if((strcmp(argv[i], "--freq") == 0))
        {
            param.proc_freq = atof(argv[++i]);
            continue;
        }
        if((strcmp(argv[i], "--vtk") == 0))
        {
            param.vtk_file = strdup(argv[++i]);
            continue;
        }
        if((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0))
        {
            printf("MD Bench: A minimalistic re-implementation of miniMD\n");
            printf(HLINE);
            printf("-f <string>:          force field (lj or eam), default lj\n");
            printf("-i <string>:          input file for EAM\n");
            printf("-n / --nsteps <int>:  set number of timesteps for simulation\n");
            printf("-nx/-ny/-nz <int>:    set linear dimension of systembox in x/y/z direction\n");
            printf("--freq <real>:        processor frequency (GHz)\n");
            printf("--vtk <string>:       VTK file for visualization\n");
            printf(HLINE);
            exit(EXIT_SUCCESS);
        }
    }

    // this should be multiple of 32 as operations are performed at the level of warps
    const int num_threads_per_block = get_num_threads();

    setup(&param, &eam, &atom, &neighbor, &c_atom, &c_neighbor, &stats, num_threads_per_block, (double*) &timer);
    computeThermo(0, &param, &atom);
    if(param.force_field == FF_EAM) {
        computeForceEam(&eam, &param, &atom, &neighbor, &stats, 1, 0);
    } else {
#if defined(MEM_TRACER) || defined(INDEX_TRACER) || defined(COMPUTE_STATS)
        computeForceTracing(&param, &atom, &neighbor, &stats, 1, 0);
#else
        computeForce(true, &param, &atom, &neighbor, &c_atom, &c_neighbor, num_threads_per_block);
#endif
    }

    timer[FORCE] = 0.0;
    timer[NEIGH] = 0.0;
    timer[TOTAL] = getTimeStamp();
    timer[NEIGH_UPDATE_ATOMS_PBC] = 0.0;
    timer[NEIGH_SETUP_PBC] = 0.0;
    timer[NEIGH_UPDATE_PBC] = 0.0;
    timer[NEIGH_BINATOMS] = 0.0;
    timer[NEIGH_BUILD_LISTS] = 0.0;

    if(param.vtk_file != NULL) {
        write_atoms_to_vtk_file(param.vtk_file, &atom, 0);
    }

    for(int n = 0; n < param.ntimes; n++) {

        const bool doReneighbour = (n + 1) % param.every == 0;

        cuda_initial_integrate(doReneighbour, &param, &atom, &c_atom, num_threads_per_block);

        if(doReneighbour) {
            timer[NEIGH] += reneighbour(&param, &atom, &neighbor, &c_atom, &c_neighbor, num_threads_per_block, (double*) &timer);
        } else {
	    double before = getTimeStamp();
            updatePbc_cuda(&atom, &param, &c_atom, false, num_threads_per_block);
	    double after = getTimeStamp();
	    timer[NEIGH_UPDATE_PBC] += after - before;
        }

        if(param.force_field == FF_EAM) {
            timer[FORCE] += computeForceEam(&eam, &param, &atom, &neighbor, &stats, 0, n + 1);
        } else {
#if defined(MEM_TRACER) || defined(INDEX_TRACER) || defined(COMPUTE_STATS)
            timer[FORCE] += computeForceTracing(&param, &atom, &neighbor, &stats, 0, n + 1);
#else
            timer[FORCE] += computeForce(doReneighbour, &param, &atom, &neighbor, &c_atom, &c_neighbor, num_threads_per_block);
#endif
        }

        cuda_final_integrate(doReneighbour, &param, &atom, &c_atom, num_threads_per_block);

        if(!((n + 1) % param.nstat) && (n+1) < param.ntimes) {
	    checkCUDAError("computeThermo atom->x memcpy back", cudaMemcpy(atom.x, c_atom.x, atom.Nmax * sizeof(MD_FLOAT) * 3, cudaMemcpyDeviceToHost) );
            computeThermo(n + 1, &param, &atom);
        }

        if(param.vtk_file != NULL) {
            write_atoms_to_vtk_file(param.vtk_file, &atom, n + 1);
        }
    }

    timer[NEIGH_BUILD_LISTS] -= timer[NEIGH_BINATOMS];
    timer[TOTAL] = getTimeStamp() - timer[TOTAL];
    computeThermo(-1, &param, &atom);

    printf(HLINE);
    printf("Force field: %s\n", ff2str(param.force_field));
    printf("Data layout for positions: %s\n", POS_DATA_LAYOUT);
#if PRECISION == 1
    printf("Using single precision floating point.\n");
#else
    printf("Using double precision floating point.\n");
#endif
    printf(HLINE);
    printf("System: %d atoms %d ghost atoms, Steps: %d\n", atom.Natoms, atom.Nghost, param.ntimes);
    printf("TOTAL %.2fs FORCE %.2fs NEIGH %.2fs REST %.2fs   NEIGH_TIMERS: UPD_AT: %.2fs SETUP_PBC %.2fs UPDATE_PBC %.2fs BINATOMS %.2fs BUILD_NEIGHBOR %.2fs\n",
            timer[TOTAL], timer[FORCE], timer[NEIGH], timer[TOTAL]-timer[FORCE]-timer[NEIGH], timer[NEIGH_UPDATE_ATOMS_PBC], timer[NEIGH_SETUP_PBC], timer[NEIGH_UPDATE_PBC], timer[NEIGH_BINATOMS], timer[NEIGH_BUILD_LISTS]);
    printf(HLINE);
    printf("Performance: %.2f million atom updates per second\n",
            1e-6 * (double) atom.Natoms * param.ntimes / timer[TOTAL]);
    double atomUpdatesTotal = (double) atom.Natoms * param.ntimes;
    printf("Force_perf in millions per sec: %.2f\n", 1e-6 * atomUpdatesTotal / timer[FORCE]);
    double atomNeighUpdatesTotal = (double) atom.Natoms * param.ntimes / param.every;
    printf("Neighbor_perf in millions per sec: updateAtomsPbc: %.2f setupPbc: %.2f updatePbc: %.2f binAtoms: %.2f buildNeighbor_wo_binning: %.2f\n", 1e-6 * atomNeighUpdatesTotal / timer[NEIGH_UPDATE_ATOMS_PBC], 1e-6 * atomNeighUpdatesTotal / timer[NEIGH_SETUP_PBC], 1e-6 * atomUpdatesTotal / timer[NEIGH_UPDATE_PBC], 1e-6 * atomNeighUpdatesTotal / timer[NEIGH_BINATOMS], 1e-6 * atomNeighUpdatesTotal / timer[NEIGH_BUILD_LISTS]);
#ifdef COMPUTE_STATS
    displayStatistics(&atom, &param, &stats, timer);
#endif
    LIKWID_MARKER_CLOSE;
    return EXIT_SUCCESS;
}
