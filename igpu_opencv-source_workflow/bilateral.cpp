#include <mathimf.h>
#include "cv_igpu_interface.hpp"
#include "igpu_comp_primitives.hpp"

#include <memory>

namespace cv {
namespace igpu {


#define bilateralFilter_atSubP(x, y) {                                                                        \
            SUM_T val[cn];                                                                                      \
            __pragma(unroll)                                                                                      \
            val[:] = cachedSrc[y + (int)space_ofs[2 * k + 1]][(x + cn * (int)space_ofs[2 * k + 1]) : cn];         \
            float diff = 0.f;                                                                                \
            unrolled_for (int c = 0; c < cn; c++)                                                              \
                diff += (float)abs(val[c] - val0[c]);                                                         \
            const float w = space_weight[k] * expf(diff * diff * gauss_color_coeff);                         \
            wsum += w;                                                                                        \
            __pragma(unroll)                                                                                  \
            sum[:] += val[:] * w;                                                                              \
        }                                                                                                     \
}


#define bilateralFilter_atPixel(x, y) {                                                                      \
        SUM_T sum[cn];                                                                                        \
        SUM_T wsum = (SUM_T)0;                                                                                 \
        __pragma(unroll)                                                                                          \
        sum[:] = (SUM_T)0;                                                                                    \
        SUM_T val0[cn];                                                                                               \
        __pragma(unroll)                                                                                             \
        val0[:] = cachedSrc[y][(cn * x) : cn];                                                                   \
        __pragma(unroll(5))                                                                                   \
        for (int k = 0; k < maxK; k++) {                                                                        \
            SUM_T val[cn];                                                                                      \
            __pragma(unroll)                                                                                      \
            val[:] = cachedSrc[y + (int)space_ofs[2 * k + 1]][(x + cn * (int)space_ofs[2 * k + 1]) : cn];         \
            float diff = 0.f;                                                                                \
            unrolled_for (int c = 0; c < cn; c++)                                                              \
                diff += (float)abs(val[c] - val0[c]);                                                         \
            const float w = space_weight[k] * expf(diff * diff * gauss_color_coeff);                         \
            wsum += w;                                                                                        \
            __pragma(unroll)                                                                                       \
            sum[:] += val[:] * w;                                                                              \
        }                                                                                                        \
        __pragma(unroll)                                                                                         \
        cachedDstRow[(cn * (x - xRadius)) : cn] = (T)(sum[:] / wsum);                                         \
}

#define computeWriteRow                                                                                                                                                                                                                                                        \
    for (int y = yRadius; y < yRadius + yTilePixSize; y++) {                                                                       \
        T cachedDstRow[xTileTSize];                                                                                               \
        for (int x = xRadius; x < xRadius + xTilePixSize; x++)                                                               \
            bilateralFilter_atPixel(x, y);                                                                                        \
                                                                                                                                    \
        /*memcpy_simple(&dst[yTile * yTileTSize + y - yRadius][xTile * xTileTSize], cachedDstRow, sizeof(cachedDstRow));*/         \
        /*__pragma(unroll)                                                                                                       \
        dst[yTile * yTileTSize + y - yRadius][(xTile * xTileTSize) : xTileTSize] = cachedDstRow[:];*/                             \
    }

template< typename T, typename SUM_T, int cn, int d, int xTilePixSize, int yTilePixSize, int yIters >
__declspec(target(gfx_kernel))
void bilateralFilter_noBorders_tiled(const uchar* restrict srcptr, int srcStep,
                                uchar* restrict dstptr, int dstStep,
                                int leftItersBorder, int rightItersBorder, int topItersBorder, int bottomItersBorder,
                                float gauss_color_coeff, int maxK, const float* restrict _space_weight, const char* restrict _space_ofs) {

    cilk_for (int yTile_ = topItersBorder; yTile_ <= bottomItersBorder; yTile_++)
        cilk_for (int xTile = leftItersBorder; xTile <= rightItersBorder; xTile++) {
            const int xWindowPixSize = d;
            const int yWindowPixSize = d;
            const int xRadius = xWindowPixSize / 2;
            const int yRadius = yWindowPixSize / 2;

            const int xTileTSize = xTilePixSize * cn;
            const int yTileTSize = yTilePixSize;
            const int xCacheTSize = (xTilePixSize + 2 * xRadius) * cn;
            const int yCacheTSize = yTilePixSize + 2 * yRadius;

            const T (* src)[srcStep] = (const T (*)[]) srcptr;
            T (* dst)[dstStep] = (T (*)[]) dstptr;

            char space_ofs[2 * (xWindowPixSize * yWindowPixSize * 0.79f + 1)];
            float space_weight[xWindowPixSize * yWindowPixSize * 0.79f + 1];

            __assume_aligned(_space_weight, 16);
            __assume_aligned(_space_ofs, 16);
            #pragma unroll
            space_weight[0 : maxK] = _space_weight[0 : maxK];
            #pragma unroll
            space_ofs[0 : 2 * maxK] = _space_ofs[0 : 2 * maxK];
            
            const int yTile = yIters * yTile_;

            atTile_initCache();
            //atTile_loadCache(xTile, yTile, cn * xRadius, yRadius);
            
            computeWriteRow;

            for (int yIter = 1; yIter < yIters; yIter++) {
                //#pragma unroll
                //cachedSrc[0 : yRadius * 2][:] = cachedSrc[(yCacheTSize - yRadius * 2) : yRadius * 2][:];
                const int yTile = yTile_ * yIters + yIter;
                //atTilePart_loadCache(xTile, yTile, cn * xRadius, yRadius, yRadius * 2, yCacheTSize - 1);

                //computeWriteRow;
            }
        }
}


struct BilateralGpuTask {
    bool async;

    Offload2dBuffer src;
    Offload2dBuffer dst;

    Offload2dBuffer space;
    
    using LayoutT = OpLayoutsServer::PrimeLayoutT;
    LayoutT layout;

    int maxK;
    int d;
    float gauss_color_coeff;

    bool tiled;

    float* space_weight;
    char* space_ofs;

    uchar stackMemory[1024];
    std::unique_ptr<uchar[]> heapMemory;
};

const int yBilatTiles = 4;

#define yTileSizes(r)           \
          ((r == 1) ? 15 :     \
           (r == 2) ? 1 :       \
           (r == 3) ? 1 :       \
           (r == 4) ? 1 :       \
           (r == 5) ? 1 :       \
           (r == 6) ? 1 :       \
           (r == 7) ? 1 : 0)

class BilateralAdviser : public Adviser {
public:
    BilateralAdviser() {
    }

    bool accept(InputArray _src, OutputArray _dst, int d,
                      double sigmaColor, double sigmaSpace,
                      int borderType) {                                                                                                   
        const int r = d / 2;

        if (d % 2 != 1)
            return false;
        if (!inRange(d, 3, 16))
            return false;
    
        if (_src.type() != CV_8UC3)  
            return false;  
        
        const int cn = 3;  
        const int depth = 1;

        float gpuPart = 1.0f;
        
        Mat src = _src.getMat();
        Mat dst = _dst.getMat();
        
        //if (src.total() * src.elemSize() < 60000000 / d || src.cols < d * 5 || src.rows * src.channels() < d * 33)
        //    return false;

        task.maxK = getMaxK(r);

        uchar* memory;
        if ((sizeof(float) + 2 * sizeof(char)) * task.maxK + 2 * 16 > sizeof(task.stackMemory)) {
            memory = new uchar[(sizeof(float) + 2 * sizeof(char)) * task.maxK + 2 * 16];
            task.heapMemory.reset(memory);
        } else
            memory = (uchar*) task.stackMemory;

        task.space_weight = (float*) (memory - ((size_t)memory) % 16 + 16);
        task.space_ofs = (char*)(task.space_weight + task.maxK);
        task.space_ofs = task.space_ofs - ((size_t)task.space_ofs) % 16 + 16;

        float gauss_color_coeff = -0.5f / (sigmaColor * sigmaColor);
        float gauss_space_coeff = -0.5f / (sigmaSpace * sigmaSpace);
        for (int i = -r, int k = 0; i <= r; i++)
            for (int j = -r; j <= r; j++) {
                float rr = std::sqrtf((float)i * i + (float)j * j);
                if (rr > r)
                    continue;
                task.space_weight[k] = (float)std::expf(rr * rr * gauss_space_coeff);
                task.space_ofs[2 * k] = i;
                task.space_ofs[2 * k++ + 1] = j;
            }
        
        if (d <= 15) {
            const int xTilePixSize = 32;
            const int yTilePixSize = yTileSizes(r) * yBilatTiles;

            c_stdout << "xTilePixSize " << xTilePixSize << " " << yTilePixSize << endl;
            
            WorkerDescriptor gpu(AtomsPerIteration(dst.elemSize() * xTilePixSize, yTilePixSize), false);
            WorkerDescriptor cpu(AtomsPerIteration(dst.elemSize(), 1), true);
        
            if (layout.accept(WorkSize(dst.elemSize() * dst.cols, dst.rows), gpuPart, gpu, cpu, cpu,
                                clarifyBordersSize(BordersSize(dst.elemSize() * r, r), src, borderType)) < 0.f)
                return false;

            c_stdout << "gpuPart " << gpuPart << endl;
            if (gpuPart < 0.f) {
                return false;
            }

            task.layout = layout.primeLayout();
            task.tiled = true;
        }/* else {
            gpuPart = layout.accept(ImageSize(src.cols, src.rows), gpuPart,
                            AtomSizeBytes(src.elemSize(), 1), AtomSizeBytes(src.elemSize(), 1),
                            ItersStart(r, r), AtomsPerIteration(1, 1),
                            LeftTopBorderSize(r, r), RightBotBorderSize(r, r));
            c_stdout << "gpuPart " << gpuPart << endl;
            if (gpuPart < 0.f) {
                return false;
            }

            task.tiled = false;
        }*/

        task.async = false;
        task.d = d;
        task.gauss_color_coeff = gauss_color_coeff;
        task.space.memoryStart = task.space.buffer = memory;
        task.space.wholeSizeBytes = (sizeof(float) + 2 * sizeof(char)) * task.maxK + 2 * 16;
        task.space.share();

        const int buffersNum = 2;
        Offload2dBuffer* buffers[buffersNum] = {&task.src, &task.dst};
        Mat* mats[buffersNum] = {&src, &dst};

        for (int i = 0; i < buffersNum; i++) {
            buffers[i]->stepBytes = mats[i]->step.p[0];
            buffers[i]->step = mats[i]->step.p[0] / mats[i]->elemSize1();
            buffers[i]->buffer = mats[i]->data;
            if (mats[i]->step.p[0] % mats[i]->elemSize1())
                return false;
            buffers[i]->memoryStart = (void*) mats[i]->datastart;
            buffers[i]->wholeSizeBytes = (size_t)(mats[i]->dataend - mats[i]->datastart);
            buffers[i]->share();
        }

        return true;
    }

    using GpuTaskT = BilateralGpuTask;
    const GpuTaskT& advisedGpu() const {
        return task;
    }
    
    //using CpuRoisT = ImageOperationLayoutCpu::ArrayT;
    //const GpuTaskT& advisedCpu() const {
    //    return task;
    //}
    //
    //using BorderRoisT = ImageBorderOperationLayout::ArrayT;
    //const BorderRoisT& advisedBorder() const {
    //    return task;
    //}

private:

    int getMaxK(int r) {
        int res = 0;
        for (int i = -r; i <= r; i++)
            for (int j = -r; j <= r; j++)
                if ((int)std::sqrtf((float)i * i + (float)j * j) <= r)
                    res++;
        return res;
    }

    BilateralGpuTask task;
};


class IgpuBilateralFilter {
public:
    IgpuBilateralFilter() {}
    
    int start(const BilateralGpuTask& _task) {
        task = &_task;
        
        const int cn = 3;

        switch(task->d) {
            case 3: return tiledInputCase<3, 32, cn>();
            case 5: return tiledInputCase<5, 32, cn>();
            //case 7: return tiledInputCase<7, 32, cn>();
            //case 9: return tiledInputCase<9, 32, cn>();
            //case 11: return tiledInputCase<11, 32, cn>();
            //case 13: return tiledInputCase<13, 32, cn>();
            //case 15: return tiledInputCase<15, 32, cn>();

            //case 17: return notTiledInputCase<17, 32, cn>();
            //case 19: return notTiledInputCase<19, 32, cn>();
            //case 21: return notTiledInputCase<21, 32, cn>();
            //case 23: return notTiledInputCase<23, 32, cn>();
        }

        return -1;
    }
    
    void finalize(int taskId) {
        if (taskId)
            _GFX_wait(taskId);
        else
            std::cerr << "finalize taskId == 0" << std::endl;
    }
    
private:

    template<int d, int xTilePixSize, int cn>
    int tiledInputCase() {
        const int r = d / 2;
        const int yTilePixSize = yTileSizes(r);
            c_stdout << "xTilePixSize " << xTilePixSize << " " << yTilePixSize << endl;
        
        Offload2dBuffer src(task->src);
        Offload2dBuffer dst(task->dst);

        Offload2dBuffer* tmp[2] = {&src, &dst};
            c_stdout << "task->layout.offset() " << task->layout.offset() << endl;
        auto workRects = prepare(task->layout.workRects(), tmp,
                task->layout.offset());

        int lastTaskId = 0;
        for (const auto& work : workRects) {
            c_stdout << "work " << work << endl;
            if (!(work.width * work.height))
                continue;
            int taskId = 1 ? 0 : _GFX_offload(&bilateralFilter_noBorders_tiled<uchar, float, cn, d, xTilePixSize, yTilePixSize, yBilatTiles>,
                                 (const uchar*)src.buffer, src.step,
                                 (uchar*)dst.buffer, dst.step,
                                 work.x, work.x + work.width - 1, work.y, work.y + work.height - 1,
                                 task->gauss_color_coeff, task->maxK, task->space_weight, task->space_ofs);
            bilateralFilter_noBorders_tiled<uchar, float, cn, d, xTilePixSize, yTilePixSize, yBilatTiles>(
                                 (const uchar*)src.buffer, src.step,
                                 (uchar*)dst.buffer, dst.step,
                                 work.x, work.x + work.width - 1, work.y, work.y + work.height - 1,
                                 task->gauss_color_coeff, task->maxK, task->space_weight, task->space_ofs);
            
            if (_GFX_get_last_error() != GFX_SUCCESS) {
                std::cerr << "_GFX_get_last_error() != GFX_SUCCESS" << std::endl;
                finalize(lastTaskId);
                throw 1;
            }

            lastTaskId = taskId;
        }

        return lastTaskId;
    }

    //template<int d, int xTilePixSize, int cn>
    //int notTiledInputCase() {
    //    const int r = d / 2;
    //    int lastK = -1;
    //    for (const auto& iters : task->itersRects) {
    //        int k = _GFX_offload(&bilateralFilter_noBorders<uchar, float, cn, d>,
    //                                (const uchar*)task->src.buffer, task->src.step,
    //                                (uchar*)task->dst.buffer, task->dst.step,
    //                                iters.x, iters.x + iters.width - 1, iters.y, iters.y + iters.height - 1,
    //                                task->gauss_color_coeff, task->maxK, task->space_weight, task->space_ofs);
    //
    //        if (k < 0) {
    //            std::c_stdout << "k < 0" << std::endl;
    //            finalize(lastK);
    //            return -1;
    //        }
    //
    //        lastK = k;
    //    }
    //
    //    return lastK;
    //}

    const BilateralGpuTask* task;
};


bool bilateralFilter( InputArray _src, OutputArray _dst, int d,
                      double sigmaColor, double sigmaSpace,
                      int borderType ) {
    Mat src = _src.getMat();

    //if (_src.isSubmatrix() || _dst.isSubmatrix())
    //    return false;

    if( sigmaColor <= 0 )
        sigmaColor = 1;
    if( sigmaSpace <= 0 )
        sigmaSpace = 1;

    if( d <= 0 )
        d = max(cvRound(sigmaSpace * 1.5), 1) * 2 + 1;

    _dst.create(src.size(), src.type());
    
    Mat dst = _dst.getMat();
        
    return performHeteroTask(BilateralAdviser(), IgpuBilateralFilter(), &cv::bilateralFilter, src, dst,
                             d, sigmaColor, sigmaSpace, borderType);
}

}
}
