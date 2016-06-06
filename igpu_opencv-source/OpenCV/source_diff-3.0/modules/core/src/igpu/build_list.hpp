#pragma once

#ifdef HAVE_IGPU
#define MINMAX_BUILD
#define BILATERAL_BUILD
#define BOXFILTER_BUILD
#define FILTER2D_BUILD
#define SEPFILTER2D_BUILD
#define GEMM_BUILD
#define CANNY_BUILD
#endif