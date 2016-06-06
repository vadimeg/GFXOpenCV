#include "../../../core/src/igpu/build_list.hpp"
#ifdef CANNY_BUILD
#include "../../../core/src/igpu/cv_igpu_interface.hpp"
#include "../../../core/src/igpu/igpu_comp_primitives.hpp"
#include "canny.hpp"

#include <memory>

namespace cv {
namespace igpu {

#define cc(y__, x__) cachedSrc[y + (y__)][x + cn * (x__)]

#define Sobel3_atPixelChannel(x, y, cachedSobelRow, addrFac, addrShift, isDx, loadSrc, storeDst) {                                                                      \
        const SUM_T sum = (isDx) ?                                                                                                                                            \
                   (cc(-1, -1) -      cc(-1, 1) +                                                                                                                        \
                    cc(0, -1) * (2) - cc(0, 1) * (2) +                                                                                                                  \
                    cc(1, -1) -       cc(1, 1)) :                                                                                                                        \
                                                                                                                                                                        \
                   (-cc(-1, -1) - cc(-1, 0) * (2) - cc(-1, 1) +                                                                                                         \
                    cc(1, -1) +   cc(1, 0) * (2) +  cc(1, 1));                                                                                                           \
                                                                                                                                                                        \
        cachedSobelRow[addrFac * (x - dstXCacheOffset) + addrShift] = (SOBEL_T) (sum + (sizeof(MAG_T) == 2 ? (32767 / 2) : 0));           \
}

#define Sobel5_atPixelChannel(x, y, cachedSobelRow, addrFac, addrShift, isDx, loadSrc, storeDst) {                                                                 \
        const SUM_T sum = (isDx) ?                                                                                                                                       \
                   (cc(-2, -2) * (1) +   cc(-2, -1) * (2) -   cc(-2, 1) * (2) -    cc(-2, 2) * (1) +                                                               \
                    cc(-1, -2) * (4) +   cc(-1, -1) * (8) -   cc(-1, 1) * (8) -    cc(-1, 2) * (4) +                                                               \
                    cc(0, -2) * (6) +    cc(0, -1) * (12) -   cc(0, 1) * (12) -     cc(0, 2) * (6) +                                                                \
                    cc(1, -2) * (4) +    cc(1, -1) * (8) -    cc(1, 1) * (8) -     cc(1, 2) * (4) +                                                                \
                    cc(2, -2) * (1) +    cc(2, -1) * (2) -    cc(2, 1) * (2) -     cc(2, 2) * (1))   :                                                              \
                                                                                                                                                                   \
                   (-cc(-2, -2) * (1) - cc(-2, -1) * (4) -  cc(-2, 0) * (6) -  cc(-2, 1) * (4) - cc(-2, 2) * (1) -                                                 \
                    -cc(-1, -2) * (2) - cc(-1, -1) * (8) -  cc(-1, 0) * (12) - cc(-1, 1) * (8) - cc(-1, 2) * (2) +                                                 \
                    cc(1, -2) * (2) +   cc(1, -1) * (8) +   cc(1, 0) * (12) +  cc(1, 1) * (8) +  cc(1, 2) * (2) +                                                  \
                    cc(2, -2) * (1) +   cc(2, -1) * (4) +   cc(2, 0) * (6) +   cc(2, 1) * (4) +  cc(2, 2) * (1));                                                   \
                                                                                                                                                                   \
        cachedSobelRow[addrFac * (x - dstXCacheOffset) + addrShift] = (SOBEL_T) (sum + (sizeof(MAG_T) == 2 ? (32767 / 2) : 0));      \
}

//#define sqrNorm(a, b) (l2Norm ? ((a) * (a) + (b) * (b)) : (a + b))
#define sqrNorm(a, b) (l2Norm ? ((a) * (a) + (b) * (b)) : (static_abs(a) + static_abs(b)))


template< typename SRC_T, typename SOBEL_T, typename MAG_T, typename KER_T, typename SUM_T, int cn, int xTilePixSize, int yTilePixSize, int aperture_size, bool l2Norm >
__declspec(target(gfx_kernel))
void Sobel_Canny_tiled(const SRC_T* restrict srcptr, int srcStep,
                                MAG_T* restrict magsPtr, int magsStep,
                                char* restrict dirsPtr, int dirsStep,
                                int xAddrLimit, int yAddrLimit,
                                int xItersStart, int xItersEnd, int yItersStart, int yItersEnd) {
    
    cilk_for (int yTile = yItersStart; yTile <= yItersEnd; yTile++)
        cilk_for (int xTile = xItersStart; xTile <= xItersEnd; xTile++) {
            const int xWindowSizePix = 3;
            const int yWindowSizePix = 3;
            const int xRadius = xWindowSizePix / 2;
            const int yRadius = yWindowSizePix / 2;
            
            const int xCachePixSize = xTilePixSize + 2 * xRadius;
            const int yCachePixSize = yTilePixSize + 2 * yRadius;
            
            SRC_T cachedSrc[yCachePixSize][cn * xCachePixSize];

            const SRC_T (* src)[srcStep] = (const SRC_T (*)[]) srcptr;
            MAG_T (* mags)[magsStep] = (MAG_T (*)[]) magsPtr;
            char (* dirs)[dirsStep] = (char (*)[]) dirsPtr;
            
            {
                const int xAddr_const = xTile * xTilePixSize - xRadius;
                int xAddr = xAddr_const;

                int len = xCachePixSize;
                    
                int leftReplicate = 0;
                int rightReplicate = 0;
                if (xAddr < 0) {
                    leftReplicate = -xAddr;
                    len -= leftReplicate;
                    xAddr = 0;
                }
                if (xAddr + len > xAddrLimit) {
                    rightReplicate = xAddr + len - xAddrLimit;
                    len -= rightReplicate;
                }
            
                if (rightReplicate >= xCachePixSize)
                    continue;
                
                if (!((yTile == yItersStart) || (yTile == yItersEnd) || (xTile == xItersStart) || (xTile == xItersEnd)))
                    cachedSrc[:][:] = src[yTile * yTilePixSize - yRadius : yCachePixSize][cn * xAddr_const : cn * xCachePixSize];
                else for (int y = 0; y < yCachePixSize; y++) {
                    const int yAddr_const = clamp(yTile * yTilePixSize + y - yRadius, 0, yAddrLimit - 1);
            
                    if (!leftReplicate && !rightReplicate) {
                        cachedSrc[y][:] = src[yAddr_const][cn * xAddr_const : cn * xCachePixSize];
                    } else {
                        cachedSrc[y][cn * leftReplicate : cn * len] = src[yAddr_const][cn * xAddr : cn * len];
                        
                        if (leftReplicate)
                            unrolled_for(int c = 0; c < cn; c++)
                                cachedSrc[y][c : leftReplicate] = cachedSrc[y][cn + c];
                        if (rightReplicate)
                            unrolled_for(int c = 0; c < cn; c++)
                                cachedSrc[y][cn * (xCachePixSize - rightReplicate) + c : rightReplicate] = cachedSrc[y][cn * (xCachePixSize - rightReplicate - 1) + c];
                    }
                }
            }
            
            for (int y = yRadius; y < yRadius + yTilePixSize; y++) {
                SOBEL_T cachedSobelRow[2 * cn * xTilePixSize];
                const int dstXCacheOffset = cn * xRadius;
               
                //cachedSobelRow[:] = 32767 / 2;
                simd_for (int x = cn * xRadius; x < cn * (xRadius + xTilePixSize); x++)
                    if (aperture_size == 3)
                        Sobel3_atPixelChannel(x, y, cachedSobelRow, 2, 0, true, loadSrc_tiled, storeDst_rowCached)
                    else if (aperture_size == 5)
                        Sobel5_atPixelChannel(x, y, cachedSobelRow, 2, 0, true, loadSrc_tiled, storeDst_rowCached)
                simd_for (int x = cn * xRadius; x < cn * (xRadius + xTilePixSize); x++)
                    if (aperture_size == 3)
                        Sobel3_atPixelChannel(x, y, cachedSobelRow, 2, 1, false, loadSrc_tiled, storeDst_rowCached)
                    else if (aperture_size == 5)
                        Sobel5_atPixelChannel(x, y, cachedSobelRow, 2, 1, false, loadSrc_tiled, storeDst_rowCached)
                

                const int xAddr = xTile * xTilePixSize;
                const int yAddr = yTile * yTilePixSize + y - yRadius;
                
                char cachedDirs[xTilePixSize];
                MAG_T cachedMags[xTilePixSize];
            
                if (sizeof(MAG_T) == 2) {
                    simd_for (int x = 0; x < xTilePixSize; x++) {
                        uint d = ((int*)cachedSobelRow)[cn * x + 0];
                    
                        int mag = sqrNorm((int)(d & 0xFFFF) - 32767 / 2, (int)((d & 0xFFFF0000) >> 16) - 32767 / 2);
                        unrolled_for (int c = 1; c < cn; c++) {
                            const uint d_ = ((int*)cachedSobelRow)[cn * x + c];
                            const int mag_ = sqrNorm((int)(d_ & 0xFFFF) - 32767 / 2, (int)((d_ & 0xFFFF0000) >> 16) - 32767 / 2);
                            if (mag_ > mag) {
                                mag = mag_;
                                d = d_;
                            }
                        }
                
                        cachedMags[x] = (MAG_T) mag;
                    
                        //in: dx, dy
                        const int dx = (int)(d & 0xFFFF) - 32767 / 2;
                        const int dy = (int)((d & 0xFFFF0000) >> 16) - 32767 / 2;
                        const float tg22Deg = 0.4142135623730950488016887242097f;
                        const float tg67Deg = 2.4142135623730950488016887242097f;
                        const float f = ((float)dy) / ((float)dx);
                        const int a_ = (int)(f * tg22Deg);
                        const int b_ = (int)(f * tg67Deg);
                        const int a = a_ != 0 ? 2 : 1;
                        const int b = b_ != 0;
                        const int dir3 = (a * b) & ((dx ^ dy) < 0); // if a = 1, b = 1, dy ^ dx < 0
                        const int dir = a * b + 2 * dir3; //dir: [0, 1, 2, 3] <-> [90, 135, 0, 45]
                        //out: dir
                
                        cachedDirs[x] = (char) (dir << 1); //dir multiplied by 2
                    }
                } else
                    simd_for (int x = 0; x < xTilePixSize; x++) {
                        MAG_T dx = cachedSobelRow[cn * 2 * x + 0];
                        MAG_T dy = cachedSobelRow[cn * 2 * x + 1];
                        
                        MAG_T mag = sqrNorm(dx, dy);
                        unrolled_for (int c = 1; c < cn; c++) {
                            MAG_T dx_ = (MAG_T)cachedSobelRow[cn * 2 * x + 2 * c];
                            MAG_T dy_ = (MAG_T)cachedSobelRow[cn * 2 * x + 2 * c + 1];
                            const MAG_T mag_ = sqrNorm(dx_, dy_);
                            if (mag_ > mag) {
                                mag = mag_;
                                dx = dx_;
                                dy = dy_;
                            }
                        }
                        
                        cachedMags[x] = (MAG_T) mag;
                        
                        //in: dx, dy
                        const float tg22Deg = 0.4142135623730950488016887242097f;
                        const float tg67Deg = 2.4142135623730950488016887242097f;
                        const float f = ((float)dy) / ((float)dx);
                        const int a_ = (int)(f * tg22Deg);
                        const int b_ = (int)(f * tg67Deg);
                        const int a = a_ != 0 ? 2 : 1;
                        const int b = b_ != 0;
                        const int dir3 = (a * b) & ((dx ^ dy) < 0); // if a = 1, b = 1, dy ^ dx < 0
                        const int dir = a * b + 2 * dir3; //dir: [0, 1, 2, 3] <-> [90, 135, 0, 45]
                        //out: dir
                        
                        cachedDirs[x] = (char) (dir << 1); //dir multiplied by 2
                    }
                
                if ((yTile + 1) <= yAddrLimit) {
                    mags[yAddr][xAddr : xTilePixSize] = cachedMags[:];
                    dirs[yAddr][xAddr : xTilePixSize] = cachedDirs[:];
                }
            }

        }
}

template< typename SRC_T, typename SOBEL_T, typename MAG_T, typename KER_T, typename SUM_T, int cn, int xTilePixSize, int yTilePixSize, int aperture_size, bool l2Norm >
void Sobel_Canny_tiled_cpu(const SRC_T* restrict srcptr, int srcStep,
                                MAG_T* restrict magsPtr, int magsStep,
                                char* restrict dirsPtr, int dirsStep,
                                int xAddrLimit, int yAddrLimit,
                                int xItersStart, int xItersEnd, int yItersStart, int yItersEnd) {
    
    cilk_for (int yTile = yItersStart; yTile <= yItersEnd; yTile++)
        cilk_for (int xTile = xItersStart; xTile <= xItersEnd; xTile++) {
            const int xWindowSizePix = 3;
            const int yWindowSizePix = 3;
            const int xRadius = xWindowSizePix / 2;
            const int yRadius = yWindowSizePix / 2;
            
            const int xCachePixSize = xTilePixSize + 2 * xRadius;
            const int yCachePixSize = yTilePixSize + 2 * yRadius;
            
            SRC_T cachedSrcLocalArray[yCachePixSize][cn * xCachePixSize];
            const SRC_T* srcCachePtr;

            const SRC_T (* src)[srcStep] = (const SRC_T (*)[]) srcptr;
            MAG_T (* mags)[magsStep] = (MAG_T (*)[]) magsPtr;
            char (* dirs)[dirsStep] = (char (*)[]) dirsPtr;

            int cachedSrcStep;
            
            if ((yTile == yItersStart) || (yTile == yItersEnd) || (xTile == xItersStart) || (xTile == xItersEnd)) {
                const int xAddr_const = xTile * xTilePixSize - xRadius;
                int xAddr = xAddr_const;

                int len = xCachePixSize;
                    
                int leftReplicate = 0;
                int rightReplicate = 0;
                if (xAddr < 0) {
                    leftReplicate = -xAddr;
                    len -= leftReplicate;
                    xAddr = 0;
                }
                if (xAddr + len > xAddrLimit) {
                    rightReplicate = xAddr + len - xAddrLimit;
                    len -= rightReplicate;
                }
            
                if (rightReplicate >= xCachePixSize)
                    continue;
                
                if (!((yTile == yItersStart) || (yTile == yItersEnd) || (xTile == xItersStart) || (xTile == xItersEnd)))
                    cachedSrcLocalArray[:][:] = src[yTile * yTilePixSize - yRadius : yCachePixSize][cn * xAddr_const : cn * xCachePixSize];
                else for (int y = 0; y < yCachePixSize; y++) {
                    const int yAddr_const = clamp(yTile * yTilePixSize + y - yRadius, 0, yAddrLimit - 1);
            
                    if (!leftReplicate && !rightReplicate) {
                        cachedSrcLocalArray[y][:] = src[yAddr_const][cn * xAddr_const : cn * xCachePixSize];
                    } else {
                        cachedSrcLocalArray[y][cn * leftReplicate : cn * len] = src[yAddr_const][cn * xAddr : cn * len];
                        
                        if (leftReplicate)
                            unrolled_for(int c = 0; c < cn; c++)
                                cachedSrcLocalArray[y][c : leftReplicate] = cachedSrcLocalArray[y][cn + c];
                        if (rightReplicate)
                            unrolled_for(int c = 0; c < cn; c++)
                                cachedSrcLocalArray[y][cn * (xCachePixSize - rightReplicate) + c : rightReplicate] = cachedSrcLocalArray[y][cn * (xCachePixSize - rightReplicate - 1) + c];
                    }
                }

                //cachedSrcLocalArray[:][:] = 0;
                cachedSrcStep = cn * xCachePixSize;
                srcCachePtr = (SRC_T*) cachedSrcLocalArray;
            }
            else {
                cachedSrcStep = srcStep;
                srcCachePtr = (SRC_T*) &src[yTile * yTilePixSize - yRadius][(xTile * xTilePixSize - xRadius) * cn];
            }

            const SRC_T (* cachedSrc)[cachedSrcStep] = (SRC_T (*)[]) srcCachePtr;
            
            for (int y = yRadius; y < yRadius + yTilePixSize; y++) {
                SOBEL_T cachedSobelRow[2 * cn * xTilePixSize];
                const int dstXCacheOffset = cn * xRadius;
               
                //cachedSobelRow[:] = 32767 / 2;
                simd_for (int x = cn * xRadius; x < cn * (xRadius + xTilePixSize); x++)
                    if (aperture_size == 3)
                        Sobel3_atPixelChannel(x, y, cachedSobelRow, 2, 0, true, loadSrc_tiled, storeDst_rowCached)
                    else if (aperture_size == 5)
                        Sobel5_atPixelChannel(x, y, cachedSobelRow, 2, 0, true, loadSrc_tiled, storeDst_rowCached)
                simd_for (int x = cn * xRadius; x < cn * (xRadius + xTilePixSize); x++)
                    if (aperture_size == 3)
                        Sobel3_atPixelChannel(x, y, cachedSobelRow, 2, 1, false, loadSrc_tiled, storeDst_rowCached)
                    else if (aperture_size == 5)
                        Sobel5_atPixelChannel(x, y, cachedSobelRow, 2, 1, false, loadSrc_tiled, storeDst_rowCached)
                

                const int xAddr = xTile * xTilePixSize;
                const int yAddr = yTile * yTilePixSize + y - yRadius;
                
                char* cachedDirs = (char*) &dirs[yAddr][xAddr];
                MAG_T* cachedMags = (MAG_T*) &mags[yAddr][xAddr];
            
                if (sizeof(MAG_T) == 2) {
                    simd_for (int x = 0; x < xTilePixSize; x++) {
                        uint d = ((int*)cachedSobelRow)[cn * x + 0];
                    
                        int mag = sqrNorm((int)(d & 0xFFFF) - 32767 / 2, (int)((d & 0xFFFF0000) >> 16) - 32767 / 2);
                        unrolled_for (int c = 1; c < cn; c++) {
                            const uint d_ = ((int*)cachedSobelRow)[cn * x + c];
                            const int mag_ = sqrNorm((int)(d_ & 0xFFFF) - 32767 / 2, (int)((d_ & 0xFFFF0000) >> 16) - 32767 / 2);
                            if (mag_ > mag) {
                                mag = mag_;
                                d = d_;
                            }
                        }
                
                        cachedMags[x] = (MAG_T) mag;
                    
                        //in: dx, dy
                        const int dx = (int)(d & 0xFFFF) - 32767 / 2;
                        const int dy = (int)((d & 0xFFFF0000) >> 16) - 32767 / 2;
                        const float tg22Deg = 0.4142135623730950488016887242097f;
                        const float tg67Deg = 2.4142135623730950488016887242097f;
                        const float f = ((float)dy) / ((float)dx);
                        const int a_ = (int)(f * tg22Deg);
                        const int b_ = (int)(f * tg67Deg);
                        const int a = a_ != 0 ? 2 : 1;
                        const int b = b_ != 0;
                        const int dir3 = (a * b) & ((dx ^ dy) < 0); // if a = 1, b = 1, dy ^ dx < 0
                        const int dir = a * b + 2 * dir3; //dir: [0, 1, 2, 3] <-> [90, 135, 0, 45]
                        //out: dir
                
                        cachedDirs[x] = (char) (dir << 1); //dir multiplied by 2
                    }
                } else
                    simd_for (int x = 0; x < xTilePixSize; x++) {
                        MAG_T dx = cachedSobelRow[cn * 2 * x + 0];
                        MAG_T dy = cachedSobelRow[cn * 2 * x + 1];
                        
                        MAG_T mag = sqrNorm(dx, dy);
                        unrolled_for (int c = 1; c < cn; c++) {
                            MAG_T dx_ = (MAG_T)cachedSobelRow[cn * 2 * x + 2 * c];
                            MAG_T dy_ = (MAG_T)cachedSobelRow[cn * 2 * x + 2 * c + 1];
                            const MAG_T mag_ = sqrNorm(dx_, dy_);
                            if (mag_ > mag) {
                                mag = mag_;
                                dx = dx_;
                                dy = dy_;
                            }
                        }
                        
                        cachedMags[x] = (MAG_T) mag;
                        
                        //in: dx, dy
                        const float tg22Deg = 0.4142135623730950488016887242097f;
                        const float tg67Deg = 2.4142135623730950488016887242097f;
                        const float f = ((float)dy) / ((float)dx);
                        const int a_ = (int)(f * tg22Deg);
                        const int b_ = (int)(f * tg67Deg);
                        const int a = a_ != 0 ? 2 : 1;
                        const int b = b_ != 0;
                        const int dir3 = (a * b) & ((dx ^ dy) < 0); // if a = 1, b = 1, dy ^ dx < 0
                        const int dir = a * b + 2 * dir3; //dir: [0, 1, 2, 3] <-> [90, 135, 0, 45]
                        //out: dir
                        
                        cachedDirs[x] = (char) (dir << 1); //dir multiplied by 2
                    }
            }

        }
}


template< int xMapCacheIntSize, int xMapTileIntSize, int yMapCacheIntSize, bool top, bool down, bool checkNext>
__declspec(target(gfx))
inline void Canny_optimizeMap(int y, uint map[][xMapCacheIntSize], const uint allRows[], uint& isNext) { 
    const int halfCellsMask = 1 | 1 << 2 | 1 << 4 | 1 << 6 | 1 << 8 |
                                1 << 10  | 1 << 12 | 1 << 14 | 1 << 16  | 1 << 18 |
                                1 << 20 | 1 << 22 | 1 << 24  | 1 << 26  | 1 << 28 |
                                1 << 30;
    const uint fullCellsMask = halfCellsMask << 1;

    if (!allRows[y]) 
        return;

    simd_for (int x = 0; x < xMapTileIntSize; x++) { //simd_for
        const uint full = map[y][x] & fullCellsMask;
        const uint neighMask = (((full << 1) | (full >> 1) | (full >> 3))) ; //left center right
        
        const uint halfCellsNeighUp = top ? map[y - 1][x] & neighMask : 0;
        if (top) {
            const uint fullCellsNeighUp = halfCellsNeighUp << 1;
            map[y - 1][x] &= ~halfCellsNeighUp;
            map[y - 1][x] |= fullCellsNeighUp;
        }
        
        const uint halfCellsNeigh = map[y][x] & neighMask;
        const uint fullCellsNeigh = halfCellsNeigh << 1;
        map[y][x] &= ~halfCellsNeigh;
        map[y][x] |= fullCellsNeigh;
        
        const uint halfCellsNeighDown = down ? map[y + 1][x] & neighMask : 0;
        if (down) {
            const uint fullCellsNeighDown = halfCellsNeighDown << 1;
            map[y + 1][x] &= ~halfCellsNeighDown;
            map[y + 1][x] |= fullCellsNeighDown;
        }
        if (checkNext)
            isNext |= halfCellsNeighUp | halfCellsNeigh | halfCellsNeighDown;
    }
}

template< typename MAG_T, int xTileTSize, int yTileTSize, int xSobelTileTSize, int ySobelTileTSize >
__declspec(target(gfx_kernel))
void localHysteresis_tiled(const MAG_T* restrict magsPtr, int magsStep,
                                const uchar* restrict dirsPtr, int dirsStep,
                                uchar* restrict flagsPtr, int flagsStep,
                                uchar* restrict dstptr, int dstStep,
                                int xAddrLimit, int yAddrLimit,
                                int xItersStart, int xItersEnd, int yItersStart, int yItersEnd,
                                MAG_T lowThreshold, MAG_T highThreshold) {
    
    cilk_for (int mapTileY = yItersStart; mapTileY <= yItersEnd; mapTileY += (yTileTSize / ySobelTileTSize))
        cilk_for (int mapTileX = xItersStart; mapTileX <= xItersEnd; mapTileX += (xTileTSize / xSobelTileTSize)) {
            const int xMapCacheBitSize = xTileTSize;
            const int yMapCacheBitSize = yTileTSize;
            const int xMapCacheIntSize = xTileTSize / (sizeof(int) * 4);
            const int yMapCacheIntSize = yTileTSize;
            
            const int xMapTileBitSize = xMapCacheBitSize;
            const int yMapTileBitSize = yMapCacheBitSize;
            const int xMapTileIntSize = xMapCacheIntSize;
            const int yMapTileIntSize = yMapCacheIntSize;
            
            const MAG_T (* mags)[magsStep] = (const MAG_T (*)[]) magsPtr;
            const uchar (* dirs)[dirsStep] = (const uchar (*)[]) dirsPtr;
            uchar (* dst)[dstStep] = (uchar (*)[]) dstptr;
            uchar (* flags)[flagsStep] = (uchar (*)[]) flagsPtr;
            
            uint map[yMapCacheIntSize][xMapCacheIntSize];
            map[:][:] = 0u;
            
            const int xSobelItersEnd = min(xTileTSize / xSobelTileTSize - 1, xItersEnd - mapTileX);
            const int ySobelItersEnd = min(yTileTSize / ySobelTileTSize - 1, yItersEnd - mapTileY);
            for (int sobelTileY = 0; sobelTileY <= ySobelItersEnd; sobelTileY++) 
                for (int sobelTileX = 0; sobelTileX <= xSobelItersEnd; sobelTileX++) {
                    const int locSupXR = 1;
                    const int locSupYR = 1;
                    const int xSobelCacheTSize = xSobelTileTSize + 2 * locSupXR;
                    const int ySobelCacheTSize = ySobelTileTSize + 2 * locSupYR;
            
                    MAG_T cachedMags[ySobelCacheTSize][xSobelCacheTSize];
            
                    const int xAddr_const = (sobelTileX + mapTileX) * xSobelTileTSize - locSupXR;
                    int xAddr = xAddr_const;
                    int len = xSobelCacheTSize;
                    
                    int leftReplicate = 0;
                    if (xAddr < 0) {
                        leftReplicate = -xAddr;
                        len -= leftReplicate;
                        xAddr = 0;
                    }
                    int rightReplicate = 0;
                    if (xAddr + len > xAddrLimit) {
                        rightReplicate = xAddr + len - xAddrLimit;
                        len -= rightReplicate;
                    }
            
                    const int yAddr_base_const = (sobelTileY + mapTileY) * ySobelTileTSize - locSupYR;
                    if (mapTileX > xItersStart && (mapTileX + xSobelItersEnd) < xItersEnd &&
                        mapTileY > yItersStart && (mapTileY + ySobelItersEnd) < yItersEnd)
                        cachedMags[:][:] = mags[yAddr_base_const : ySobelCacheTSize][xAddr_const : xSobelCacheTSize];
                    else for (int y = 0; y < ySobelCacheTSize; y++) {
                        const int yAddr_const = yAddr_base_const + y;
                        
                        if (!leftReplicate && !rightReplicate && yAddr_const >= 0 && yAddr_const < yAddrLimit)
                            cachedMags[y][:] = mags[yAddr_const][xAddr_const : xSobelCacheTSize];
                        else if (yAddr_const < 0 || yAddr_const >= yAddrLimit)
                            cachedMags[y][:] = -1;
                        else {
                            cachedMags[y][leftReplicate : len] = mags[yAddr_const][xAddr : len];

                            if (leftReplicate)
                                cachedMags[y][0] = -1;
                            if (rightReplicate)
                                cachedMags[y][xSobelCacheTSize - rightReplicate] = -1;
                        }
                    }
                    
                    for (int y = 0; y < ySobelTileTSize; y++) { //0-80% of execution time
                        const int notTooLow = __sec_reduce_any_nonzero(cachedMags[locSupYR + y][locSupXR : xSobelTileTSize] > lowThreshold);
            
                        if (notTooLow) {
                            const uint prevX_arrAsBitmap = 0 | 0 | (1 << 4) | 0; //0 <-> -1, 1 <-> 0, 2 <-> 1
                            const uint prevY_arrAsBitmap = 1 | (2 << 2) | 0 | 0;
                            const uint nextX_arrAsBitmap = 2 | (2 << 2) | (1 << 4) | (2 << 6);
                            const uint nextY_arrAsBitmap = 1 | 0 | (2 << 4) | (2 << 6);
                            
                            uchar cachedDirs_types[xSobelTileTSize];
                            cachedDirs_types[:] = dirs[(sobelTileY + mapTileY) * ySobelTileTSize + y]
                                                      [(sobelTileX + mapTileX) * xSobelTileTSize : xSobelTileTSize];
                            
                            const int y_ = y + locSupYR;

                            int prevs[xSobelTileTSize];
                            int nexts[xSobelTileTSize];

                            simd_for (int x = 0; x < xSobelTileTSize; x++) {
                                const int x_ = locSupXR + x;
                                prevs[x] = x_ + (int)((prevX_arrAsBitmap >> cachedDirs_types[x]) & 3) - 1;
                                nexts[x] = x_ + (int)((nextX_arrAsBitmap >> cachedDirs_types[x]) & 3) - 1;
                            }
                            prevs[:] += (y_ + (int)((prevY_arrAsBitmap >> cachedDirs_types[:]) & 3) - 1) * xSobelCacheTSize;
                            nexts[:] += (y_ + (int)((nextY_arrAsBitmap >> cachedDirs_types[:]) & 3) - 1) * xSobelCacheTSize;

                            MAG_T prevMags[xSobelTileTSize];
                            MAG_T nextMags[xSobelTileTSize];

                            prevMags[:] = cachedMags[0][prevs[:]]; //0-25%  of execution time
                            nextMags[:] = cachedMags[0][nexts[:]] + (MAG_T)((cachedDirs_types[:] & 2) >> 1); //0-25%  of execution time

                            cachedDirs_types[:] = cachedMags[y_][1 : xSobelTileTSize] > prevMags[:] && cachedMags[y_][1 : xSobelTileTSize] >= nextMags[:];
                            
                            cachedDirs_types[:] = cachedDirs_types[:] ?
                                ((cachedMags[y_][locSupXR : xSobelTileTSize] > highThreshold) + (cachedMags[y_][locSupXR : xSobelTileTSize] > lowThreshold)) : 0;
                            
                            unrolled_simd_for (int x = 0; x < static_max(xSobelTileTSize / 4, 32); x++) //0-20% of execution time
                                unrolled_for (int b = 0; b < 8; b += 2)
                                    if (x < xSobelTileTSize / 4)
                                        ((uchar*)(&map[sobelTileY * ySobelTileTSize + y]
                                                      [sobelTileX * (xSobelTileTSize / (sizeof(int) * 4))]))[x] |= (uchar) (cachedDirs_types[x * 4 + b / 2] << b);
                        }
                    }
                }
            
            const uint halfCellsMask = 1 | 1 << 2 | 1 << 4 | 1 << 6 | 1 << 8 |
                                        1 << 10  | 1 << 12 | 1 << 14 | 1 << 16  | 1 << 18 |
                                        1 << 20 | 1 << 22 | 1 << 24  | 1 << 26  | 1 << 28 |
                                        1 << 30;
            const uint fullCellsMask = halfCellsMask << 1;
            
            uint flagsLocal[xMapCacheIntSize];
            uint all = 0;

            {
                uint allCols[xMapCacheIntSize];
                allCols[:] = 0;
                
                for (int y = 0; y < yMapTileIntSize; y++)
                    allCols[:] |= map[y][:];
            
                all = __sec_reduce_or(allCols[:]);
            
                if (all & halfCellsMask) {
                    flagsLocal[:] = ((allCols[:] & 1) | (allCols[:] & (1 << 30)) |
                                    ((map[0][:] | map[yMapTileIntSize - 1][:]) & halfCellsMask)) != 0;
            
                    flags[mapTileY / (yTileTSize / ySobelTileTSize)]
                        [mapTileX * (xMapTileIntSize / (xTileTSize / xSobelTileTSize)) : xMapCacheIntSize] = (uchar)flagsLocal[:];
                } else
                    flagsLocal[:] = 0;
            }
            
            const bool nothingToCompute = !(all & halfCellsMask) || !(all & fullCellsMask);
            
            if (!nothingToCompute) {
                uint allRows[yMapCacheIntSize];
            
                for (int y = 0; y < yMapTileIntSize; y++)
                    allRows[y] = __sec_reduce_or(map[y][:]);
            
                uint isNext = 1;
            
                int start = 0;
                int end = yMapCacheIntSize - 1;
                for (; !allRows[start]; start++);
                for (; !allRows[end]; end--);
            
                while (isNext) {
                    isNext = 0;
                    
                    Canny_optimizeMap<xMapCacheIntSize, xMapTileIntSize, yMapCacheIntSize, false, true, true>(start, map, allRows, isNext);
            
                    for (int y = start + 1; y <= end - 1; y++)
                        if (isNext)
                            Canny_optimizeMap<xMapCacheIntSize, xMapTileIntSize, yMapCacheIntSize, true, true, false>(y, map, allRows, isNext);
                        else
                            Canny_optimizeMap<xMapCacheIntSize, xMapTileIntSize, yMapCacheIntSize, true, true, true>(y, map, allRows, isNext);
                    
                    if (isNext)
                        Canny_optimizeMap<xMapCacheIntSize, xMapTileIntSize, yMapCacheIntSize, true, false, false>(end, map, allRows, isNext);
                    else
                        Canny_optimizeMap<xMapCacheIntSize, xMapTileIntSize, yMapCacheIntSize, true, false, true>(end, map, allRows, isNext);
                    
                    if (!isNext)
                        break;
                    
                    for (int y = end - 1; y >= start + 1; y--)
                        if (isNext)
                            Canny_optimizeMap<xMapCacheIntSize, xMapTileIntSize, yMapCacheIntSize, true, true, false>(y, map, allRows, isNext);
                        else
                            Canny_optimizeMap<xMapCacheIntSize, xMapTileIntSize, yMapCacheIntSize, true, true, true>(y, map, allRows, isNext);
                }
            }
            
            const int xAddr = mapTileX * xSobelTileTSize;
            int len = xTileTSize;
                
            bool rightReplicate = false;
            if (xAddr + len > xAddrLimit) {
                len = xAddrLimit - xAddr;
                rightReplicate = true;
            }
            
            if (all)
                for (int y = 0; y < yMapTileBitSize; y++) { //writing
                    const int yAddr = mapTileY * ySobelTileTSize + y;
                    if (yAddr >= yAddrLimit)
                        break;

                    uint imgRow[xMapCacheIntSize * sizeof(int)];
                    simd_for (int x = 0; x < xMapTileIntSize; x++) //simd_for
                        unrolled_for (int i = 0; i < sizeof(int); i++) { //bytes
                            const uint bitmap = 0x0 | 0x180 << 8 | 0xFF << 16;

                            uint m0;
                            uint m1;
                            uint m2;
                            uint m3;
                            //{
                            //    const uint offset = i * 8;
                            //    const uint m_ = (map[y][x] >> offset) & 3;
                            //    m0 = (m_ * 120);
                            //}
                            //{
                            //    const uint offset = i * 8 + 2;
                            //    const uint m_ = (map[y][x] >> offset) & 3;
                            //    m1 = (m_ * 120) << 8;
                            //}
                            //{
                            //    const uint offset = i * 8 + 4;
                            //    const uint m_ = (map[y][x] >> offset) & 3;
                            //    m2 = (m_ * 120) << 16;
                            //}
                            //{
                            //    const uint offset = i * 8 + 6;
                            //    const uint m_ = (map[y][x] >> offset) & 3;
                            //    m3 = (m_ * 120) << 24;
                            //}
                            {
                                const uint offset = i * 8;
                                const uint m_ = (map[y][x] >> offset) & 3;
                                m0 = ((m_ == 2) ? 255u : (m_ & flagsLocal[x]));
                                //if (m_ == 1 && !flagsLocal[x])
                                //    m0 = 180;
                            }
                            {
                                const uint offset = i * 8 + 2;
                                const uint m_ = (map[y][x] >> offset) & 3;
                                m1 = ((m_ == 2) ? 255u : (m_ & flagsLocal[x])) << 8;
                                //if (m_ == 1 && !flagsLocal[x])
                                //    m0 = 180 << 8;
                            }
                            {
                                const uint offset = i * 8 + 4;
                                const uint m_ = (map[y][x] >> offset) & 3;
                                m2 = ((m_ == 2) ? 255u : (m_ & flagsLocal[x])) << 16;
                                //if (m_ == 1 && !flagsLocal[x])
                                //    m0 = 180 << 16;
                            }
                            {
                                const uint offset = i * 8 + 6;
                                const uint m_ = (map[y][x] >> offset) & 3;
                                m3 = ((m_ == 2) ? 255u : (m_ & flagsLocal[x])) << 24;
                                //if (m_ == 1 && !flagsLocal[x])
                                //    m0 = 180 << 24;
                            }
                            
                            imgRow[x * sizeof(int) + i] = m0 | m1 | m2 | m3;
                        }

                    //if (!y)
                    //    imgRow[:] |= 135 | 123 << 8 | 123 << 16 | 123 << 24;
                    //imgRow[0] |= 135;
                
                    if (!rightReplicate)
                        dst[yAddr][xAddr : xTileTSize] = ((uchar*)imgRow)[0 : xTileTSize];
                    else
                        dst[yAddr][xAddr : len] = ((uchar*)imgRow)[0 : len];
                    //dst[yAddr][xAddr : !rightReplicate ? xTileTSize : len] = ((uchar*)imgRow)[0 : !rightReplicate ? xTileTSize : len];
                }
            else
                for (int y = 0; y < yMapTileBitSize; y++) {
                    const int yAddr = mapTileY * ySobelTileTSize + y;
                    if (yAddr >= yAddrLimit)
                        break;
                    if (!rightReplicate)
                        dst[yAddr][xAddr : xTileTSize] = 0;
                    else
                        dst[yAddr][xAddr : len] = 0;
                }
        }
}

#define pushIfHalfCell_(i, x, y) {                   \
    if (img[y][x] == (uchar) 1) {                    \
        img[y][x] = (uchar) 255;                     \
        localStacks[i][stackSizes[i]] = x;           \
        localStacks[i][stackSizes[i] + 1] = y;       \
        stackSizes[i] += 2;                          \
    }                                                \
}
#define pushIfHalfCell(i, x, y) {                    \
    const int y_ = clamp(y, 0, rows - 1);            \
    const int x_ = clamp(x, 0, cols - 1);            \
    pushIfHalfCell_(i, x_, y_);                      \
}

#define pushHalfCells_(i, x, y)  {                   \
    if (img[y][x] == (uchar) 255) {                  \
        pushIfHalfCell(i, x - 1, y - 1);             \
        pushIfHalfCell(i, x, y - 1);                 \
        pushIfHalfCell(i, x + 1, y - 1);             \
        pushIfHalfCell(i, x - 1, y);                 \
        pushIfHalfCell(i, x + 1, y);                 \
        pushIfHalfCell(i, x - 1, y + 1);             \
        pushIfHalfCell(i, x, y + 1);                 \
        pushIfHalfCell(i, x + 1, y + 1);             \
    }                                                \
}

#define pushHalfCells(i, x, y)  {                    \
    const int y__ = clamp(y, 0, rows - 1);           \
    const int x__ = clamp(x, 0, cols - 1);           \
    pushHalfCells_(i, x__, y__);                     \
}

template <int xTileTSize, int yTileTSize, int parallelLevel>
void hysteresisImg_global(uchar* restrict img_, int imgStep, int cols, int rows, int* restrict stack, int stackStep, const uchar* restrict flags_, int flagsStep) {
    uchar (* flags)[flagsStep] = (uchar (*)[])(flags_);
    uchar (* img)[imgStep] = (uchar (*)[])(img_);
    
    int yFlagsIters = div(rows, yTileTSize, true);
    int yFlagsThreadIters = div(yFlagsIters, parallelLevel, true);
    int xFlagsIters = div(cols, xTileTSize, true);
    cilk_for (int thread = 0; thread < parallelLevel; thread++)
        for (int row_ = 0; row_ < yFlagsThreadIters; row_++) {
            const int row = thread * yFlagsThreadIters + row_;
            if (row >= yFlagsIters)
                break;

            int* localStacks[2] = {stack + stackStep * 2 * thread,
                                   stack + stackStep * (2 * thread + 1)};
            int stackSizes[2] = {0, 0};

            //int i = 0;
            //for (int col = 0; col < xFlagsIters;) {
            //    for (; col < xFlagsIters && !flags[row][col]; col++);
            //    if (col >= xFlagsIters)
            //        break;
            //
            //    for (int y = row * yTileTSize - 1; y <= row * yTileTSize; y++)
            //        for (int x = col * xTileTSize - 1; x <= (col + 1) * xTileTSize; x++)
            //            pushHalfCells(i, x, y);
            //    for (int y = row * yTileTSize + 1; y <= (row + 1) * xTileTSize - 2; y++) {
            //        pushHalfCells(i, col * xTileTSize - 1, y);
            //        pushHalfCells(i, col * xTileTSize, y);
            //        pushHalfCells(i, (col + 1) * xTileTSize - 1, y);
            //        pushHalfCells(i, (col + 1) * xTileTSize, y);
            //    }
            //    for (int y = (row + 1) * yTileTSize - 1; y <= (row + 1) * yTileTSize; y++)
            //        for (int x = col * xTileTSize - 1; x <= (col + 1) * xTileTSize; x++)
            //            pushHalfCells(i, x, y);
            //
            //    col++;
            //}
            //
            //while (stackSizes[0] || stackSizes[1]) {
            //    for (int s = 0; s < stackSizes[i]; s += 2) {
            //        pushHalfCells(!i, localStacks[i][s], localStacks[i][s + 1]);
            //    }
            //    stackSizes[i] = 0;
            //    i = !i;
            //}

            for (int col = 0; col < xFlagsIters;) {
                for (; col < xFlagsIters && !flags[row][col]; col++);
                const int left = col;
                for (; col < xFlagsIters && flags[row][col]; col++);
                const int right = col;
                
                if (left == right)
                    continue;

                int i = 0;
                for (int y = row * yTileTSize - 1; y <= row * yTileTSize; y++)
                    for (int x = left * xTileTSize - 1; x <= right * xTileTSize; x++)
                        pushHalfCells(i, x, y);
                for (int y = row * yTileTSize + 1; y <= (row + 1) * xTileTSize - 2; y++) {
                    pushHalfCells(i, left * xTileTSize - 1, y);
                    pushHalfCells(i, left * xTileTSize, y);

                    for (int x_center = left; x_center < right; x_center++) {
                        pushHalfCells(i, x_center * xTileTSize - 1, y);
                        pushHalfCells(i, x_center * xTileTSize, y);
                        pushHalfCells(i, x_center * xTileTSize + 1, y);
                    }

                    pushHalfCells(i, right * xTileTSize - 1, y);
                    pushHalfCells(i, right * xTileTSize, y);
                }
                for (int y = (row + 1) * yTileTSize - 1; y <= (row + 1) * yTileTSize; y++)
                    for (int x = left * xTileTSize - 1; x <= right * xTileTSize; x++)
                        pushHalfCells(i, x, y);
                while (stackSizes[0] || stackSizes[1]) {
                    for (int s = 0; s < stackSizes[i]; s += 2) {
                        pushHalfCells(!i, localStacks[i][s], localStacks[i][s + 1]);
                    }
                    stackSizes[i] = 0;
                    i = !i;
                }
            }
        }
    
    //cilk_for (int y = 0; y < rows; y++) 
    //    simd_for (int x = 0; x < cols; x++)
    //        if (img[y][x] == 1)
    //            img[y][x] = 0;
    cilk_for (int thread = 0; thread < parallelLevel; thread++)
        for (int row_ = 0; row_ < yFlagsThreadIters; row_++) {
            const int row = thread * yFlagsThreadIters + row_;
            if (row >= yFlagsIters)
                break;
            
            const int bot_addr = min(row * (yTileTSize + 1), rows);
            for (int col = 0; col < xFlagsIters;) {
                for (; col < xFlagsIters && !flags[row][col]; col++);
                const int left = col;
                for (; col < xFlagsIters && flags[row][col]; col++);
                const int right = col;
                
                if (left != right)
                    for (int y = row * yTileTSize; y < bot_addr; y++) {
                        const int right_addr = min(right * xTileTSize, cols);
                        simd_for (int x = left * xTileTSize; x < right_addr; x++)
                            if (img[y][x] == 1)
                                img[y][x] = 0;
                    }
            }
        }
}


struct CannyGpuTask {
    bool async;

    Offload2dBuffer src;
    Offload2dBuffer dirs;
    Offload2dBuffer mags;

    Offload2dBuffer flags;
    Offload2dBuffer dst;
    //int ddepth;

    using PrimeLayoutT = OpLayoutsServer::PrimeLayoutT;
    using SecondLayoutT = OpLayoutsServer::SecondLayoutT;
    PrimeLayoutT gpuSobelLayout;
    SecondLayoutT cpuSobelLayout;

    PrimeLayoutT gpuCannyLayout;
    SecondLayoutT cpuCannyLayout;

    int cols;
    int rows;

    bool l2;
    int lowThreshold;
    int highThreshold;
    int apertureSize;

    int xAddrLimit;
    int yAddrLimit;

    int cn;
};

class CannyAdviser : public Adviser {
public:

    using GpuTaskT = CannyGpuTask;
    const GpuTaskT& advisedGpu() const {
        return task;
    }

    CannyAdviser() {}

    bool accept(InputArray _src, OutputArray _dst, double low_thresh, double high_thresh,
               int aperture_size, bool L2gradient) {

        if (_src.type() != CV_8UC3 && _src.type() != CV_8UC1)  
            return false;

        if (!(aperture_size % 2))
            return false;

        if (!inRange(aperture_size, 3, 3))
            return false;
                                     
        const int cn = _src.channels();  
        
        //t2start
        Mat src = _src.getMat();
        Mat dst = _dst.getMat();
        //t2end("getMat() ")

        if (src.datastart == dst.datastart)
            return false;

        bool is16bitMag = !L2gradient && (aperture_size == 3);
        
        {
            const int xTileTSize = cn == 3 ? 64 : 128;
            const int yTileTSize = cn == 3 ? 5 : 8;
            
            WorkerDescriptor gpu(AtomsPerIteration(xTileTSize, yTileTSize), true);
            WorkerDescriptor cpu(AtomsPerIteration(xTileTSize, yTileTSize), true);
        
            if (!layout.accept(WorkSize(dst.cols, dst.rows), getGpuPart("IGPU_CANNY_SOBEL_WORK_PART"), gpu, cpu))
                return false;
        
            task.gpuSobelLayout = layout.primeLayout();
            task.cpuSobelLayout = layout.secondLayout();
            //c_stdout << "task.gpuSobelLayout " << task.gpuSobelLayout.workRects()[0] << endl;
            //c_stdout << "task.cpuSobelLayout " << task.cpuSobelLayout.workRects()[0] << endl;
        }
        
        {
            const int xTileTSize = is16bitMag ? 32 : 32;
            const int yTileTSize = is16bitMag ? 16 : 8;
        
            WorkerDescriptor gpu(AtomsPerIteration(xTileTSize, yTileTSize), true);
            WorkerDescriptor cpu(AtomsPerIteration(xTileTSize, yTileTSize), true);
        
            if (!layout.accept(WorkSize(dst.cols, dst.rows), getGpuPart("IGPU_CANNY_HYST_WORK_PART"), gpu, cpu))
                return false;
        
            task.gpuCannyLayout = layout.primeLayout();
            task.cpuCannyLayout = layout.secondLayout();
            //c_stdout << "task.gpuCannyLayout " << task.gpuCannyLayout.workRects()[0] << endl;
            //c_stdout << "task.cpuCannyLayout " << task.cpuCannyLayout.workRects()[0] << endl;
        }
        //t2end("layout ")
        
        Size s = dst.size();
        s.width = (div(s.width, 128, true) + 1) * 128;
        s.height = (div(s.height, 10, true) + 1) * 10;
        s.height = (div(s.height, 16, true) + 1) * 16;
        mags.create(s, is16bitMag ? CV_16SC1 : CV_32SC1);
        dirs.create(s, CV_8UC1);
        s.width /= 16;
        s.height /= 16;
        flags = Mat::zeros(s, CV_8UC1);
        
        task.async = false;
        
        const int buffersNum = 5;
        Offload2dBuffer* buffers[buffersNum] = {&task.src, &task.dst, &task.mags, &task.dirs, &task.flags};
        Mat* mats[buffersNum] = {&src, &dst, &mags, &dirs, &flags};
        
        for (int i = 0; i < buffersNum; i++) {
            *buffers[i] = Offload2dBuffer(*mats[i]);
            if (!buffers[i]->share())
                return false;
        }
        
        task.cols = dst.cols;
        task.rows = dst.rows;
        task.apertureSize = aperture_size;
        task.cn = cn;

        task.l2 = L2gradient;
        
        task.xAddrLimit = dst.cols;
        task.yAddrLimit = dst.rows;

        if (task.l2) {
            high_thresh *= high_thresh;
            low_thresh *= low_thresh;
        }
        if (is16bitMag) {
            task.highThreshold = (std::numeric_limits<short>::max() / 2) <= high_thresh ?
                                                                    (std::numeric_limits<short>::max() / 2) : cvRound(high_thresh);
            task.lowThreshold = (std::numeric_limits<short>::max() / 2) <= low_thresh ?
                                                                    (std::numeric_limits<short>::max() / 2) : cvRound(low_thresh);
        } else {
            task.highThreshold = (std::numeric_limits<int>::max()) <= high_thresh ?
                                                                    (std::numeric_limits<short>::max() / 2) : cvRound(high_thresh);
            task.lowThreshold = (std::numeric_limits<int>::max()) <= low_thresh ?
                                                                    (std::numeric_limits<short>::max() / 2) : cvRound(low_thresh);
        }
        
        //t2end("create ")
        
        return true;
    }
    
private:
    Mat flags;
    Mat mags;
    Mat dirs;

    CannyGpuTask task;
};


class CannyWorker {
public:
    CannyWorker() {}

    void start(const CannyGpuTask& _task) {
        task = &_task;
        
        int tIdS = 0;
        
        //t2start
        if (task->cn == 3) {
            if (task->apertureSize == 3) {
                const int cn = 3;
                if (!task->l2) {
                    const bool l2 = false;
                    tIdS = start_SobelGpu<short, 64, 5, cn, 3, l2>();
                    start_SobelCpu<short, 64, 5, cn, 3, l2>();
                } else {
                    const bool l2 = true;
                    tIdS = start_SobelGpu<int, 64, 5, cn, 3, l2>();
                    start_SobelCpu<int, 64, 5, cn, 3, l2>();
                }
            }
        } else if (task->cn == 1) {
            if (task->apertureSize == 3) {
                const int cn = 1;
                if (!task->l2) {
                    const bool l2 = false;
                    tIdS = start_SobelGpu<short, 128, 8, cn, 3, l2>();
                    start_SobelCpu<short, 128, 8, cn, 3, l2>();
                } else {
                    const bool l2 = true;
                    tIdS = start_SobelGpu<int, 128, 8, cn, 3, l2>();
                    start_SobelCpu<int, 128, 8, cn, 3, l2>();
                }
            }
        } else
		    throw std::runtime_error("CannyWorker start: unsupported cn");
        
        if (task->apertureSize == 3 && !task->l2) {
            start_hysteresisGpu<short, 256, 16, 32, 16>();
            finalize(tIdS);
            start_hysteresisCpu<short, 256, 16, 32, 16>();
        } else {
            start_hysteresisGpu<int, 256, 16, 32, 8>();
            finalize(tIdS);
            //c_stdout << "int" << endl;
            start_hysteresisCpu<int, 256, 16, 32, 8>();
        }
        
        const int hysteresisParallelLevel = 64;
        int* stacks;
        std::unique_ptr<int[]> scopeDeletable;
        if (task->mags.wholeSizeBytes >= (2 * hysteresisParallelLevel * 2 * (task->cols + task->rows) * sizeof(int)))
            stacks = (int*) task->mags.memoryStart;
        else
            scopeDeletable.reset(stacks = new int[2 * hysteresisParallelLevel * 2 * (task->cols + task->rows)]);
        
        hysteresisImg_global<16, 16, hysteresisParallelLevel>((uchar*) task->dst.buffer, task->dst.step, task->cols, task->rows,
                        stacks, 2 * (task->cols + task->rows), (const uchar*) task->flags.buffer, task->flags.step);
        
    }

    void finalize() {
        finalize(lastTaskId);

        //c_stdout << "finalize taskId " << lastTaskId << std::endl;

    }
    
private:

    void finalize(GfxTaskId taskId) {
        if (taskId && _GFX_wait(taskId))
		    throw std::runtime_error("CannyWorker finalize: _GFX_get_last_error() != GFX_SUCCESS");
        
        if (taskId == lastTaskId)
            lastTaskId = 0;
        //c_stdout << "finalize taskId " << taskId << std::endl;
    }
    
    GfxTaskId lastTaskId = 0;

    template<typename MAG_T, int xTileTSize, int yTileTSize, int cn, int aperture_size, bool l2Norm>
    int start_SobelGpu() {
        Offload2dBuffer src(task->src);
        Offload2dBuffer mags(task->mags);
        Offload2dBuffer dirs(task->dirs);
        
        shiftWithOffset(task->gpuSobelLayout.offset(), src, mags, dirs);
        auto workRects = task->gpuSobelLayout.workRects();
        for (const auto& work : workRects) {
            if (!(work.width * work.height))
                continue;
            //c_stdout << "work " << work << " yTileTSize " << yTileTSize << endl;
            GfxTaskId taskId = _GFX_offload(&Sobel_Canny_tiled<uchar, short, MAG_T, short, short, cn, xTileTSize, yTileTSize, aperture_size, l2Norm>,
                                    (const uchar*)src.buffer, src.step,
                                    (MAG_T*)mags.buffer, mags.step,
                                    (char*)dirs.buffer, dirs.step,
                                    task->xAddrLimit, task->yAddrLimit,
                                    work.x, work.x + work.width - 1, work.y, work.y + work.height - 1);
            
            if (_GFX_get_last_error() != GFX_SUCCESS) {
                finalize();
		        throw std::runtime_error("CannyWorker start: _GFX_get_last_error() != GFX_SUCCESS");
            }

            lastTaskId = taskId;
        }

        return lastTaskId;
    }

    template<typename MAG_T, int xTileTSize, int yTileTSize, int cn, int aperture_size, bool l2Norm>
    void start_SobelCpu() {
        Offload2dBuffer src(task->src);
        Offload2dBuffer mags(task->mags);
        Offload2dBuffer dirs(task->dirs);
        
        shiftWithOffset(task->cpuSobelLayout.offset(), src, mags, dirs);
        auto workRects = task->cpuSobelLayout.workRects();
        for (const auto& work : workRects) {
            if (!(work.width * work.height))
                continue;
            //c_stdout << "work " << work << " yTileTSize " << yTileTSize << endl;
            Sobel_Canny_tiled_cpu<uchar, short, MAG_T, short, short, cn, xTileTSize, yTileTSize, aperture_size, l2Norm>(
                                    (const uchar*)src.buffer, src.step,
                                    (MAG_T*)mags.buffer, mags.step,
                                    (char*)dirs.buffer, dirs.step,
                                    task->xAddrLimit, task->yAddrLimit,
                                    work.x, work.x + work.width - 1, work.y, work.y + work.height - 1);
        }
    }

    template<typename MAG_T, int xTileTSize, int yTileTSize, int xSobelTileTSize, int ySobelTileTSize>
    int start_hysteresisGpu() {
        Offload2dBuffer dst(task->dst);
        Offload2dBuffer mags(task->mags);
        Offload2dBuffer dirs(task->dirs);
        Offload2dBuffer flags(task->flags);
        
        shiftWithOffset(task->gpuCannyLayout.offset(), mags, dirs, flags, dst);
        auto workRects = task->gpuCannyLayout.workRects();
        for (const auto& work : workRects) {
            if (!(work.width * work.height))
                continue;
            //c_stdout << "work " << work << " yTileTSize " << yTileTSize << " ySobelTileTSize " << ySobelTileTSize << endl;
            GfxTaskId taskId = _GFX_offload(&localHysteresis_tiled<MAG_T, xTileTSize, yTileTSize, xSobelTileTSize, ySobelTileTSize>,
                                    (const MAG_T*)mags.buffer, mags.step,
                                    (const uchar*)dirs.buffer, dirs.step,
                                    (uchar*)flags.buffer, flags.step,
                                    (uchar*)dst.buffer, dst.step,
                                    task->xAddrLimit, task->yAddrLimit,
                                    work.x, work.x + work.width - 1, work.y, work.y + work.height - 1,
                                    (MAG_T)task->lowThreshold, (MAG_T)task->highThreshold);

            if (_GFX_get_last_error() != GFX_SUCCESS) {
                finalize();
		        throw std::runtime_error("CannyWorker start: _GFX_get_last_error() != GFX_SUCCESS");
            }

            lastTaskId = taskId;
        }

        return lastTaskId;
    }

    template<typename MAG_T, int xTileTSize, int yTileTSize, int xSobelTileTSize, int ySobelTileTSize>
    void start_hysteresisCpu() {
        Offload2dBuffer dst(task->dst);
        Offload2dBuffer mags(task->mags);
        Offload2dBuffer dirs(task->dirs);
        Offload2dBuffer flags(task->flags);
        
        shiftWithOffset(task->cpuCannyLayout.offset(), mags, dirs, flags, dst);
        auto workRects = task->cpuCannyLayout.workRects();
        for (const auto& work : workRects) {
            if (!(work.width * work.height))
                continue;
            //c_stdout << "work " << work << " yTileTSize " << yTileTSize << " ySobelTileTSize " << ySobelTileTSize << endl;
            localHysteresis_tiled<MAG_T, xTileTSize, yTileTSize, xSobelTileTSize, ySobelTileTSize>(
                                    (const MAG_T*)mags.buffer, mags.step,
                                    (const uchar*)dirs.buffer, dirs.step,
                                    (uchar*)flags.buffer, flags.step,
                                    (uchar*)dst.buffer, dst.step,
                                    task->xAddrLimit, task->yAddrLimit,
                                    work.x, work.x + work.width - 1, work.y, work.y + work.height - 1,
                                    (MAG_T)task->lowThreshold, (MAG_T)task->highThreshold);
        }
    }

    const CannyGpuTask* task;
};



bool Canny( InputArray _src, OutputArray _dst,
               double low_thresh, double high_thresh,
               int aperture_size, bool L2gradient ) {
    Mat src = _src.getMat();
    _dst.create(src.size(), CV_8UC1);
    Mat dst = _dst.getMat();
        
    {
        CannyAdviser adviser;
        //t2start
        if (!adviser.accept(src, dst, low_thresh, high_thresh, aperture_size, L2gradient))
            return false;
        //t2end("    adviser           ")

        const auto& gpuTask = adviser.advisedGpu();

        if (!gpuTask.async) {
        
            CannyWorker worker;
            worker.start(gpuTask);
        
        //t2end("    start           ")

            worker.finalize();
        }
    }

    return true;
}


}
}
#endif
