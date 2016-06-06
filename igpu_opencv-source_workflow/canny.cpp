#include "cv_igpu_interface.hpp"
#include "igpu_comp_primitives.hpp"
#include "filter2d.hpp"

#include <memory>

namespace cv {
namespace igpu {


#define Sobel3_atPixelChannal(x, y, cachedSobelRow, addrFac, addrShift, isDx, loadSrc, storeDst) {                        \
        SUM_T sum;                                                                                                         \
        if (isDx)                                                                                                     \
            sum = cachedSrc[y - 1][x - cn] - cachedSrc[y - 1][x + cn] +                                      \
                  cachedSrc[y][x - cn] * (2) -   cachedSrc[y][x + cn] * (2) +                                          \
                  cachedSrc[y + 1][x - cn] - cachedSrc[y + 1][x + cn];                                       \
        else                                                                                                          \
            sum = -cachedSrc[y - 1][x - cn] - cachedSrc[y - 1][x] * (2) - cachedSrc[y - 1][x + cn] +     \
                  cachedSrc[y + 1][x - cn] +     cachedSrc[y + 1][x] * (2) +    cachedSrc[y + 1][x + cn];          \
                                                                                                                      \
        cachedSobelRow[addrFac * (x - dstXCacheOffset) + addrShift] = (SOBEL_T) (sum + (sizeof(MAG_T) == 2 ? (std::numeric_limits<SOBEL_T>::max() / 2) : 0)); \
}

//#define sqrNorm(a, b) (l2Norm ? ((a) * (a) + (b) * (b)) : (a + b))
#define sqrNorm(a, b) (l2Norm ? ((a) * (a) + (b) * (b)) : (static_abs(a) + static_abs(b)))


template< typename SRC_T, typename SOBEL_T, typename MAG_T, typename KER_T, typename SUM_T, bool cpu, int cn, int xTilePixSize, int yTilePixSize, bool l2Norm >
__declspec(target(gfx_kernel))
void Sobel3_Canny_tiled(const SRC_T* restrict srcptr, int srcStep,
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
            
            if (!cpu || yTile == yItersStart || yTile == yItersEnd || xTile == xItersStart || xTile == xItersEnd) {
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
                
                if (!(yTile == yItersStart || yTile == yItersEnd || xTile == xItersStart || xTile == xItersEnd))
                    cachedSrcLocalArray[:][:] = src[yTile * yTilePixSize - yRadius : yCachePixSize][cn * xAddr_const : cn * xCachePixSize];
                else for (int y = 0; y < yCachePixSize; y++) {
                    const int yAddr_const = clamp(yTile * yTilePixSize + y - yRadius, 0, yAddrLimit - 1);
            
                    if (!leftReplicate && !rightReplicate) {
                        cachedSrcLocalArray[y][:] = src[yAddr_const][cn * xAddr_const : cn * xCachePixSize];
                        //memcpy_(&cachedSrcLocalArray[y][0],
                        //        &src[yAddr_const][cn * xAddr_const],
                        //        cn * xCachePixSize * sizeof(SRC_T));
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
                
                unrolled_simd_for (int x = cn * xRadius; x < cn * (xRadius + xTilePixSize); x++)
                    Sobel3_atPixelChannal(x, y, cachedSobelRow, 2, 0, true, loadSrc_tiled, storeDst_rowCached)
                unrolled_simd_for (int x = cn * xRadius; x < cn * (xRadius + xTilePixSize); x++)
                    Sobel3_atPixelChannal(x, y, cachedSobelRow, 2, 1, false, loadSrc_tiled, storeDst_rowCached)
                

                const int xAddr = xTile * xTilePixSize;
                const int yAddr = yTile * yTilePixSize + y - yRadius;
                
                char cachedDirs_[xTilePixSize];
                MAG_T cachedMags_[xTilePixSize];
                char* cachedDirs = cpu ? (char*) &dirs[yAddr][xAddr] : (char*) cachedDirs_;
                MAG_T* cachedMags = cpu ? (MAG_T*) &mags[yAddr][xAddr] : (MAG_T*) cachedMags_;
            
                if (sizeof(MAG_T) == 2) {
                    simd_for (int x = 0; x < xTilePixSize; x++) {
                        uint d = ((int*)cachedSobelRow)[cn * x + 0];
                    
                        int mag = sqrNorm((int)(d & 0xFFFF) - std::numeric_limits<SOBEL_T>::max() / 2, (int)((d & 0xFFFF0000) >> 16) - std::numeric_limits<SOBEL_T>::max() / 2);
                        unrolled_for (int c = 1; c < cn; c++) {
                            const uint d_ = ((int*)cachedSobelRow)[cn * x + c];
                            const int mag_ = sqrNorm((int)(d_ & 0xFFFF) - std::numeric_limits<SOBEL_T>::max() / 2, (int)((d_ & 0xFFFF0000) >> 16) - std::numeric_limits<SOBEL_T>::max() / 2);
                            if (mag_ > mag) {
                                mag = mag_;
                                d = d_;
                            }
                        }
                
                        cachedMags[x] = (MAG_T) mag;
                    
                        //in: dx, dy
                        const int dx = (int)(d & 0xFFFF) - std::numeric_limits<SOBEL_T>::max() / 2;
                        const int dy = (int)((d & 0xFFFF0000) >> 16) - std::numeric_limits<SOBEL_T>::max() / 2;
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
                
                if (!cpu && (yTile + 1) <= yAddrLimit) {
                    mags[yAddr][xAddr : xTilePixSize] = cachedMags_[:];
                    dirs[yAddr][xAddr : xTilePixSize] = cachedDirs_[:];
                    //memcpy_(&mags[yAddr][xAddr],
                    //        cachedMags,
                    //        xTilePixSize * sizeof(MAG_T));
                    //
                    //memcpy_(&dirs[yAddr][xAddr],
                    //        cachedDirs,
                    //        xTilePixSize);
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
        const uint neighMask = ((full << 1) | (full >> 1) | (full >> 3)); //left center right
        
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
void Canny_tiled(const MAG_T* restrict magsPtr, int magsStep,
                                const uchar* restrict dirsPtr, int dirsStep,
                                uchar* restrict flagsPtr, int flagsStep,
                                uchar* restrict dstptr, int dstStep,
                                int xAddrLimit, int yAddrLimit,
                                int xItersStart, int xItersEnd, int yItersStart, int yItersEnd,
                                MAG_T lowThreshold, MAG_T highThreshold) {
    
    cilk_for (int mapTileY = yItersStart; mapTileY <= yItersEnd; mapTileY += (yTileTSize / ySobelTileTSize))
        cilk_for (int mapTileX = xItersStart; mapTileX <= xItersEnd; mapTileX += (xTileTSize / xSobelTileTSize)) {
            using SobelT = MAG_T;
            const int depth = sizeof(char);
            
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
            map[:][:] = 0;
            
            const int xSobelItersEnd = min(xTileTSize / xSobelTileTSize - 1, xItersEnd - mapTileX);
            const int ySobelItersEnd = min(yTileTSize / ySobelTileTSize - 1, yItersEnd - mapTileY);
            for (int sobelTileY = 0; sobelTileY <= ySobelItersEnd; sobelTileY++) 
                for (int sobelTileX = 0; sobelTileX <= xSobelItersEnd; sobelTileX++) {
                    const int localSupXRadius = 1;
                    const int localSupYRadius = 1;
                    const int xSobelCacheTSize = xSobelTileTSize + 2 * localSupXRadius;
                    const int ySobelCacheTSize = ySobelTileTSize + 2 * localSupYRadius;
            
                    MAG_T cachedMags[ySobelCacheTSize][xSobelCacheTSize];
            
                    const int xAddr_const = (sobelTileX + mapTileX) * xSobelTileTSize - localSupXRadius;
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
            
                    for (int y = 0; y < ySobelCacheTSize; y++) {
                        const int yAddr_const = clamp((sobelTileY + mapTileY) * ySobelTileTSize + y - localSupYRadius, 0, yAddrLimit - 1);
            
                        if (!leftReplicate && !rightReplicate)
                            cachedMags[y][:] = mags[yAddr_const][xAddr_const : xSobelCacheTSize];
                        else {
                            cachedMags[y][leftReplicate : len] = mags[yAddr_const][xAddr : len];

                            if (leftReplicate)
                                cachedMags[y][0] = cachedMags[y][1];
                            if (rightReplicate)
                                cachedMags[y][xSobelCacheTSize - rightReplicate] = cachedMags[y][xSobelCacheTSize - rightReplicate - 1];
                        }
                    }
                    
                    for (int y = 0; y < ySobelTileTSize; y++) {
                        int notTooLow = 0;
                        unrolled_simd_for (int x = 0; x < xSobelTileTSize; x++)
                            notTooLow |= cachedMags[localSupYRadius + y][localSupXRadius + x] > lowThreshold;
            
                        if (notTooLow) {
                            const uint prevX_arrAsBitmap = 0 | 0 | (1 << 4) | 0; //0 <-> -1, 1 <-> 0, 2 <-> 1
                            const uint prevY_arrAsBitmap = 1 | (2 << 2) | 0 | 0;
                            const uint nextX_arrAsBitmap = 2 | (2 << 2) | (1 << 4) | (2 << 6);
                            const uint nextY_arrAsBitmap = 1 | 0 | (2 << 4) | (2 << 6);
                            
                            uchar cachedDirs_types[xSobelTileTSize];
                            cachedDirs_types[:] = dirs[(sobelTileY + mapTileY) * ySobelTileTSize + y]
                                                      [(sobelTileX + mapTileX) * xSobelTileTSize : xSobelTileTSize];

                            //uchar types[xSobelTileTSize];
                            unrolled_simd_for (int x = 0; x < xSobelTileTSize; x++) { //simd_for
                                const MAG_T mag = cachedMags[localSupYRadius + y][localSupXRadius + x];
                                const uint dir = (uint) cachedDirs_types[x];
                            
                                //dir multiplyed by 2
                                const MAG_T prevMag = cachedMags[localSupYRadius + y + (int)((prevY_arrAsBitmap >> dir) & 3) - 1]
                                                                [localSupXRadius + x + (int)((prevX_arrAsBitmap >> dir) & 3) - 1];
                                const MAG_T nextMag = cachedMags[localSupYRadius + y + (int)((nextY_arrAsBitmap >> dir) & 3) - 1]
                                                                [localSupXRadius + x + (int)((nextX_arrAsBitmap >> dir) & 3) - 1] + (MAG_T)((dir & 2) >> 1);
                            
                                const bool isLocalMax = mag > prevMag && mag >= nextMag;
                            
                                cachedDirs_types[x] = isLocalMax ? ((mag > highThreshold) + (mag > lowThreshold)) : 0;
                            }

                            unrolled_simd_for (int x = 0; x < static_max(xSobelTileTSize / 4, 32); x++)
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
            
            uint allRows[yMapCacheIntSize];
            uint allCols[xMapCacheIntSize];
            uint all = 0;
            
            {
                unrolled_simd_for (int x = 0; x < xMapTileIntSize; x++)
                    allRows[0] = map[0][x];
                unrolled_for (int y = 1; y < yMapTileIntSize; y++)
                    unrolled_simd_for (int x = 0; x < xMapTileIntSize; x++) //simd_for 
                        allRows[y] |= map[y][x];
                unrolled_simd_for (int x = 0; x < xMapTileIntSize; x++) //simd_for 
                    allCols[x] = map[0][x];
                unrolled_for (int y = 0; y < yMapTileIntSize; y++)
                    unrolled_simd_for (int x = 0; x < xMapTileIntSize; x++) //simd_for 
                        allCols[x] |= map[y][x];
            
                simd_for (int y = 0; y < xMapCacheIntSize; y++)
                    all |= allCols[y];
            }
            
            const bool nothingToCompute = !(all & halfCellsMask) || !(all & fullCellsMask);
            
            if (!nothingToCompute) {
            
                allRows[:] &= fullCellsMask;
            
                uint isNext = 1;
            
                int start = 0;
                int end = yMapCacheIntSize - 1;
                for (; !allRows[start]; start++);
                for (; !allRows[end]; end--);
            
                while (isNext) {
                    isNext = 0;
                    
                    Canny_optimizeMap<xMapCacheIntSize, xMapTileIntSize, yMapCacheIntSize, false, true, true>(start, map, allRows, isNext);
            
                    unrolled_for (int y = start + 1; y <= end - 1; y++)
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
                    
                    unrolled_for (int y = end - 1; y >= start + 1; y--)
                        if (isNext)
                            Canny_optimizeMap<xMapCacheIntSize, xMapTileIntSize, yMapCacheIntSize, true, true, false>(y, map, allRows, isNext);
                        else
                            Canny_optimizeMap<xMapCacheIntSize, xMapTileIntSize, yMapCacheIntSize, true, true, true>(y, map, allRows, isNext);
                }
            }
            
            uint flagsLocal[xMapCacheIntSize];
            
            {
                //uchar flagsLocal_c[xMapCacheIntSize];
            
                if (all & halfCellsMask) {
                    flagsLocal[:] = ((allCols[:] & 1) | (allCols[:] & (1 << 30)) |
                                    ((map[0][:] | map[yMapTileIntSize - 1][:]) & halfCellsMask)) != 0;
                    //flagsLocal_c[:] = flagsLocal[:];
            
                    flags[mapTileY / (yTileTSize / ySobelTileTSize)]
                        [mapTileX * (xMapTileIntSize / (xTileTSize / xSobelTileTSize)) : xMapCacheIntSize] = (uchar)flagsLocal[:];
                } else
                    flagsLocal[:] = 0;
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
                    for (int x = 0; x < xMapTileIntSize; x++) //simd_for
                        unrolled_for (int i = 0; i < sizeof(int); i++) { //bytes
                            uint m[4];
                            const uint bitmap = 0x0 | 0x1 << 8 | 0xFF << 16;
                            //const uchar bitmap[3] = {0x0, 0x1, 0xFF};
                            unrolled_for (int bit = 0; bit < 8; bit += 2) { //bits
                                //const uint m_ = map[y][x] & (3 << (i * 8 + bit));
                                const uint offset = i * 8 + bit;
                                const uint m_ = (map[y][x] & (3 << offset)) >> offset;
                                //m[bit/2] = (m_ == 2 ? 235u : m_ * flagsLocal[x] * 50u) + flagsLocal[x] * 20u;
                                //m[bit/2] = ((m_ == 2) ? 255u : (m_ & flagsLocal[x]));
                                //m[bit/2] = (bitmap >> (m_ * 8)) & 0xFF;
                                m[bit / 2] = ((uchar*)&bitmap)[m_];

                                //if (m_ == 2)
                                //    m[bit / 2] = 255;
                                //else if (m_ == 1)
                                //    m[bit / 2] = flagsLocal[x];
                                //else
                                //    m[bit / 2] = 0;
                            }
                            imgRow[x * sizeof(int) + i] = m[0] | m[1] << 8 | m[2] << 16 | m[3] << 24;
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
void hystImg(uchar* restrict img_, int imgStep, int cols, int rows, int* restrict stack, int stackStep, const uchar* restrict flags_, int flagsStep) {
    uchar (* flags)[flagsStep] = (uchar (*)[])(flags_);
    uchar (* img)[imgStep] = (uchar (*)[])(img_);
    
    int yFlagsIters = div(rows, yTileTSize, true);
    int yFlagsThreadIters = div(yFlagsIters, parallelLevel, true);
    int xFlagsIters = div(cols, xTileTSize, true);
    cilk_for (int thread = 0; thread < parallelLevel; thread++)
        for (int flagsRow_ = 0; flagsRow_ < yFlagsThreadIters; flagsRow_++) {
            const int flagsRow = thread * yFlagsThreadIters + flagsRow_;
            if (flagsRow >= yFlagsIters)
                break;

            int* localStacks[2] = {stack + stackStep * 2 * thread,
                                   stack + stackStep * (2 * thread + 1)};
            int stackSizes[2] = {0, 0};

            for (int flagsCol = 0; flagsCol < xFlagsIters;) {
                for (; flagsCol < xFlagsIters && !flags[flagsRow][flagsCol]; flagsCol++);
                const int left = flagsCol;
                for (; flagsCol < xFlagsIters && flags[flagsRow][flagsCol]; flagsCol++);
                const int right = flagsCol;
                
                if (left != right)
                    for (int y = flagsRow * yTileTSize - 1; y < flagsRow * yTileTSize + yTileTSize + 1; y++) {
                        int i = 0;
                        for (int x = left * xTileTSize - 1; x < right * xTileTSize + 1; x++)
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
        }

    cilk_for (int thread = 0; thread < parallelLevel; thread++)
        for (int flagsRow_ = 0; flagsRow_ < yFlagsThreadIters; flagsRow_++) {
            const int flagsRow = thread * yFlagsThreadIters + flagsRow_;
            if (flagsRow >= yFlagsIters)
                break;

            for (int flagsCol = 0; flagsCol < xFlagsIters;) {
                for (; flagsCol < xFlagsIters && !flags[flagsRow][flagsCol]; flagsCol++);
                const int left = flagsCol;
                for (; flagsCol < xFlagsIters && flags[flagsRow][flagsCol]; flagsCol++);
                const int right = flagsCol;

                if (left != right)
                    for (int y = flagsRow * yTileTSize; y < flagsRow * yTileTSize + yTileTSize; y++) {
                        simd_for (int x = left * xTileTSize; x < right * xTileTSize; x++)
                            if (img[y][x] == 1)
                                img[y][x] = 0;
                            //img[y][x] = (((img[y][x] & 2) >> 1) | 254) & img[y][x];
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
                                     
        const int cn = _src.channels();  
        
        float gpuPart = 0.0f;
        
        t2start
        Mat src = _src.getMat();
        Mat dst = _dst.getMat();
        t2end("getMat() ")

        bool is16bitMag = !L2gradient && (aperture_size == 3);
        
        {
            const int xTileTSize = cn == 3 ? 64 : 128;
            const int yTileTSize = cn == 3 ? 5 : 8;
            
            WorkerDescriptor gpu(AtomsPerIteration(xTileTSize, yTileTSize), true);
            WorkerDescriptor cpu(AtomsPerIteration(xTileTSize, yTileTSize), true);
        
            if (!layout.accept(WorkSize(dst.cols, dst.rows), 0.75, gpu, cpu))
                return false;
        
            task.gpuSobelLayout = layout.primeLayout();
            task.cpuSobelLayout = layout.secondLayout();
            c_stdout << "task.gpuSobelLayout " << task.gpuSobelLayout.workRects()[0] << endl;
            c_stdout << "task.cpuSobelLayout " << task.cpuSobelLayout.workRects()[0] << endl;
        }
        
        {
            const int xTileTSize = is16bitMag ? 32 : 32;
            const int yTileTSize = is16bitMag ? 16 : 8;
        
            WorkerDescriptor gpu(AtomsPerIteration(xTileTSize, yTileTSize), true);
            WorkerDescriptor cpu(AtomsPerIteration(xTileTSize, yTileTSize), true);
        
            if (!layout.accept(WorkSize(dst.cols, dst.rows), 0.75, gpu, cpu))
                return false;
        
            task.gpuCannyLayout = layout.primeLayout();
            task.cpuCannyLayout = layout.secondLayout();
            c_stdout << "task.gpuCannyLayout " << task.gpuCannyLayout.workRects()[0] << endl;
            c_stdout << "task.cpuCannyLayout " << task.cpuCannyLayout.workRects()[0] << endl;
        }
        t2end("layout ")
        
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
            buffers[i]->buffer = mats[i]->data;
            buffers[i]->stepBytes = mats[i]->step.p[0];
            buffers[i]->step = mats[i]->step.p[0] / mats[i]->elemSize1();
            if (mats[i]->step.p[0] % mats[i]->elemSize1())
                return false;
            buffers[i]->memoryStart = (void*) mats[i]->datastart;
            buffers[i]->wholeSizeBytes = (size_t)(mats[i]->dataend - mats[i]->datastart);
            buffers[i]->share();
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
        
        t2end("create ")
        
        return true;
    }

    CannyGpuTask task;
    
private:
    Mat flags;
    Mat mags;
    Mat dirs;
};


class CannyCpuGpuWorker {
public:
    CannyCpuGpuWorker() {}

    int start(const CannyGpuTask& _task) {
        task = &_task;
        
        int tIdS = 0;
        int tIdC = 0;

        if (task->cn == 3) {
            if (task->apertureSize == 3) {
                const int cn = 3;
                if (!task->l2) {
                    const bool l2 = false;
                    tIdS = inputCaseSobelGpu<short, 64, 5, cn, l2>();
                    inputCaseSobelCpu<short, 64, 5, cn, l2>();
                } else {
                    const bool l2 = true;
                    tIdS = inputCaseSobelGpu<int, 64, 5, cn, l2>();
                    inputCaseSobelCpu<int, 64, 5, cn, l2>();
                }
            }
        } else if (task->cn == 1) {
            if (task->apertureSize == 3) {
                const int cn = 1;
                if (!task->l2) {
                    const bool l2 = false;
                    tIdS = inputCaseSobelGpu<short, 128, 8, cn, l2>();
                    inputCaseSobelCpu<short, 128, 8, cn, l2>();
                }
            }
        }

        if (task->apertureSize == 3 && !task->l2) {
            tIdC = inputCaseCannyGpu<short, 256, 16, 32, 16>();
            finalize(tIdS);
            inputCaseCannyCpu<short, 256, 16, 32, 16>();
        } else {
            tIdC = inputCaseCannyGpu<int, 256, 16, 32, 8>();
            finalize(tIdS);
            c_stdout << "int" << endl;
            inputCaseCannyCpu<int, 256, 16, 32, 8>();
        }
        hystImg<16, 16, 4>((uchar*) task->dst.buffer, task->dst.step, task->cols, task->rows,
                        (int*) task->mags.buffer, task->mags.step * 8, (const uchar*) task->flags.buffer, task->flags.step);
        
        return tIdC;
    }

    void finalize(int taskId) {
        if (taskId)
            _GFX_wait(taskId);
        else
            c_stdout << "finalize taskId == 0" << std::endl;

    }
    
private:

    template<typename MAG_T, int xTileTSize, int yTileTSize, int cn, bool l2Norm>
    int inputCaseSobelGpu() {
        Offload2dBuffer src(task->src);
        Offload2dBuffer mags(task->mags);
        Offload2dBuffer dirs(task->dirs);

        Offload2dBuffer* tmp[3] = {&src, &mags, &dirs};
            //c_stdout << "task->gpuSobelLayout.offset() " << task->gpuSobelLayout.offset() << endl;
        std::vector<Rect> workRects = prepare(task->gpuSobelLayout.workRects(), tmp,
                task->gpuSobelLayout.offset());
        
        //c_stdout << "task->src.buffer" << task->src.buffer << endl;
        //c_stdout << "src.buffer" << src.buffer << endl;
        //c_stdout << "src.step" << src.step << endl;
        //c_stdout << "task->mags.buffer" << task->mags.buffer << endl;
        //c_stdout << "mags.buffer" << mags.buffer << endl;
        //c_stdout << "mags.step" << mags.step << endl;
        //c_stdout << "task->dirs.buffer" << task->dirs.buffer << endl;
        //c_stdout << "dirs.buffer" << dirs.buffer << endl;
        //c_stdout << "dirs.step" << dirs.step << endl;

        int lastTaskId = 0;
        for (const auto& work : workRects) {
            if (!(work.width * work.height))
                continue;
            c_stdout << "work " << work << " yTileTSize " << yTileTSize << endl;
            int taskId = _GFX_offload(&Sobel3_Canny_tiled<uchar, short, MAG_T, short, short, false, cn, xTileTSize, yTileTSize, l2Norm>,
                                    (const uchar*)src.buffer, src.step,
                                    (MAG_T*)mags.buffer, mags.step,
                                    (char*)dirs.buffer, dirs.step,
                                    task->xAddrLimit, task->yAddrLimit,
                                    work.x, work.x + work.width - 1, work.y, work.y + work.height - 1);
            
            if (_GFX_get_last_error() != GFX_SUCCESS) {
                std::cerr << "_GFX_get_last_error() != GFX_SUCCESS" << std::endl;
                finalize(lastTaskId);
                throw 1;
            }

            lastTaskId = taskId;
        }

        return lastTaskId;
    }

    template<typename MAG_T, int xTileTSize, int yTileTSize, int cn, bool l2Norm>
    void inputCaseSobelCpu() {
        Offload2dBuffer src(task->src);
        Offload2dBuffer mags(task->mags);
        Offload2dBuffer dirs(task->dirs);

        Offload2dBuffer* tmp[3] = {&src, &mags, &dirs};
        std::vector<Rect> workRects = prepare(task->cpuSobelLayout.workRects(), tmp,
                task->gpuSobelLayout.offset());

        for (const auto& work : workRects) {
            if (!(work.width * work.height))
                continue;
            c_stdout << "work " << work << " yTileTSize " << yTileTSize << endl;
            Sobel3_Canny_tiled<uchar, short, MAG_T, short, short, true, cn, xTileTSize, yTileTSize, l2Norm>(
                                    (const uchar*)src.buffer, src.step,
                                    (MAG_T*)mags.buffer, mags.step,
                                    (char*)dirs.buffer, dirs.step,
                                    task->xAddrLimit, task->yAddrLimit,
                                    work.x, work.x + work.width - 1, work.y, work.y + work.height - 1);
        }
    }

    template<typename MAG_T, int xTileTSize, int yTileTSize, int xSobelTileTSize, int ySobelTileTSize>
    int inputCaseCannyGpu() {
        Offload2dBuffer dst(task->dst);
        Offload2dBuffer mags(task->mags);
        Offload2dBuffer dirs(task->dirs);
        Offload2dBuffer flags(task->flags);

        Offload2dBuffer* tmp[4] = {&dst, &mags, &dirs, &flags};
        std::vector<Rect> workRects = prepare(task->gpuCannyLayout.workRects(), tmp,
                task->gpuSobelLayout.offset());
        
        //c_stdout << "task->dst.buffer" << task->dst.buffer << endl;
        //c_stdout << "dst.buffer" << dst.buffer << endl;
        //c_stdout << "dst.step" << dst.step << endl;
        //c_stdout << "task->mags.buffer" << task->mags.buffer << endl;
        //c_stdout << "mags.buffer" << mags.buffer << endl;
        //c_stdout << "mags.step" << mags.step << endl;
        //c_stdout << "task->dirs.buffer" << task->dirs.buffer << endl;
        //c_stdout << "dirs.buffer" << dirs.buffer << endl;
        //c_stdout << "dirs.step" << dirs.step << endl;
        //c_stdout << "task->flags.buffer" << task->flags.buffer << endl;
        //c_stdout << "flags.buffer" << flags.buffer << endl;
        //c_stdout << "flags.step" << flags.step << endl;

        int lastTaskId = 0;
        for (const auto& work : workRects) {
            if (!(work.width * work.height))
                continue;
            c_stdout << "work " << work << " yTileTSize " << yTileTSize << endl;
            int taskId = _GFX_offload(&Canny_tiled<MAG_T, xTileTSize, yTileTSize, xSobelTileTSize, ySobelTileTSize>,
                                    (const MAG_T*)mags.buffer, mags.step,
                                    (const uchar*)dirs.buffer, dirs.step,
                                    (uchar*)flags.buffer, flags.step,
                                    (uchar*)dst.buffer, dst.step,
                                    task->xAddrLimit, task->yAddrLimit,
                                    work.x, work.x + work.width - 1, work.y, work.y + work.height - 1,
                                    (MAG_T)task->lowThreshold, (MAG_T)task->highThreshold);

            if (_GFX_get_last_error() != GFX_SUCCESS) {
                std::cerr << "_GFX_get_last_error() != GFX_SUCCESS" << std::endl;
                finalize(lastTaskId);
                throw 1;
            }

            lastTaskId = taskId;
        }

        return lastTaskId;
    }

    template<typename MAG_T, int xTileTSize, int yTileTSize, int xSobelTileTSize, int ySobelTileTSize>
    void inputCaseCannyCpu() {
        Offload2dBuffer dst(task->dst);
        Offload2dBuffer mags(task->mags);
        Offload2dBuffer dirs(task->dirs);
        Offload2dBuffer flags(task->flags);

        Offload2dBuffer* tmp[4] = {&dst, &mags, &dirs, &flags};
        std::vector<Rect> workRects = prepare(task->cpuCannyLayout.workRects(), tmp,
                task->gpuSobelLayout.offset());

        for (const auto& work : workRects) {
            if (!(work.width * work.height))
                continue;
            c_stdout << "work " << work << " yTileTSize " << yTileTSize << " ySobelTileTSize " << ySobelTileTSize << endl;
            Canny_tiled<MAG_T, xTileTSize, yTileTSize, xSobelTileTSize, ySobelTileTSize>(
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

    tstart
        
    CannyAdviser adviser;
    t2start
    if (!adviser.accept(src, dst, low_thresh, high_thresh, aperture_size, L2gradient))
        return false;
    t2end("    adviser             ")

    const auto& gpuTask = adviser.advisedGpu();

    if (!gpuTask.async) {
        
        CannyCpuGpuWorker worker;
        int k = worker.start(gpuTask);
        if (k == 0) {
            //c_stdout << "k == 0" << endl;
        }
        
    t2end("    start             ")

        worker.finalize(k);
    t2end("    wait             ")
        
    tend("    ALL             ")
    tflush
    }

    return true;
}


}
}
