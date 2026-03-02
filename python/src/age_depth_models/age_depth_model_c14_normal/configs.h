#ifndef CONFIGS_H
#define CONFIGS_H

#include <stdint.h>
#include <stdbool.h>

/* ---------------- HMCConfig ---------------- */

typedef struct
{
    int64_t ndt; /* number of time steps */
    int64_t nch; /* number of chains */
    int64_t ns;  /* number of samples */
    int64_t sd;  /* seed */

    double dt; /* time step */

    double *sp; /* starting points */
} HMCConfig;

/* ---------------- Data ---------------- */

typedef struct
{
    int64_t N;      /* number of sedimentation rates */
    int64_t nc14;   /* number of c14 data points */
    int64_t nd18o;  /* number of d18o data points */
    int64_t nd18or; /* number of d18o reference values */

    double H;  /* sediment depth */
    double dc; /* segment depth */
    double pm; /* prior mu */
    double ps; /* prior sigma */
    double th; /* theta */

    double *cs;     /* discrete depth points */
    double *c14;    /* c14 ages */
    double *c14d;   /* c14 depths */
    double *c14s;   /* c14 sigma */
    double *ic14v;  /* c14 inverse variance */
    double *d18o;   /* d18o values */
    double *d18od;  /* d18o depths */
    double *d18os;  /* d18o sigma */
    double *id18ov; /* d18o inverse variance */
    double *d18or;  /* d18o reference values */
    double *d18ort; /* d18o reference times */
    double *d18ota; /* d18o true ages */

    char *dn; /* data name */
} Data;

#endif /* CONFIGS_H */
