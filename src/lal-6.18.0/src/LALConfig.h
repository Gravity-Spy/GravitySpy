/* src/LALConfig.h.  Generated from LALConfig.h.in by configure.  */
/* only include this file if LAL's config.h has not been included */
#ifndef LAL_VERSION

/* LAL Version */
#define LAL_VERSION "6.18.0"

/* LAL Version Major Number  */
#define LAL_VERSION_MAJOR 6

/* LAL Version Minor Number  */
#define LAL_VERSION_MINOR 18

/* LAL Version Micro Number  */
#define LAL_VERSION_MICRO 0

/* LAL Version Devel Number  */
#define LAL_VERSION_DEVEL 0

/* Suppress debugging code */
/* #undef LAL_NDEBUG */

/* Use functions rather than macros */
/* #undef NOLALMACROS */

/* Use pthread mutex lock for threadsafety */
#define LAL_PTHREAD_LOCK 1

/* Define if using fftw3 library */
#define LAL_FFTW3_ENABLED 1

/* Define if using fftw3 aligned memory optimizations */
/* #undef LAL_FFTW3_MEMALIGN_ENABLED */

/* Define if using boinc library */
/* #undef LAL_BOINC_ENABLED */

/* Define if using CUDA library */
/* #undef LAL_CUDA_ENABLED */

/* Define if using HDF5 library */
#define LAL_HDF5_ENABLED 1

/* Define if using qthread library */
/* #undef LAL_QTHREAD */

#endif /* LAL_VERSION */
