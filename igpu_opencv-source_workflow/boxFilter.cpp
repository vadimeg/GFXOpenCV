#include "cv_igpu_interface.hpp"
#include "igpu_comp_primitives.hpp"
#include "boxFilter.hpp"

namespace cv {
namespace igpu {

#define asResult(res) ((T)(normalized ? ((res) / factor) : clamp((res) / factor, minSum, maxSum)))

#define computeHardRow(source)                                                                                                              \
            unrolled_for (int c = 0; c < cn; c++) {                                                                                                    \
                SUM_T slidingRes = source[c];                                                                                                       \
                for (int i = 1; i < xWindowSizePix; i++)                                                                                       \
                    slidingRes += source[c + cn * i];                                                                                               \
                                                                                                                                                \
                cachedDstRow[c] = asResult(slidingRes);                                                                                          \
                                                                                                                                                \
                for (int xColRes = cn * xWindowSizePix + c; xColRes < xCacheTSize; xColRes += cn) {                                         \
                    slidingRes += source[xColRes] - source[xColRes - cn * xWindowSizePix];                                                          \
                    cachedDstRow[xColRes - cn * xWindowSizePix + cn] = asResult(slidingRes);                                                    \
                }                                                                                                                               \
            }                                     

#define computeEasyRow(source)                                                                                                                    \
            simd_for (int xColRes = cn * xRadius; xColRes < cn * xRadius + xTileTSize; xColRes++) {                                                        \
                SUM_T res = (SUM_T) source[xColRes - cn * xRadius];                                                          \
                unrolled_for (int xColSum = -cn * (xRadius - 1); xColSum <= cn * xRadius; xColSum += cn)                                                           \
                    res += (SUM_T) source[xColRes + xColSum];                                                                                                      \
                cachedDstRow[xColRes - cn * xRadius] = asResult(res);                                                                               \
            }

#define computeWriteRow(source, yRes) {                                                                                                                                 \
            T cachedDstRow[xCacheTSize];                                                                                                              \
            if (isEasy)                                                                                                                       \
                computeEasyRow(source)                                                                                                             \
            else                                                                                                                                          \
                computeHardRow(source)                                                                                                             \
            __pragma(unroll)                                                 \
            dst[yTile * yTileTSize + (yRes)][(xTile * xTileTSize) : xTileTSize] = cachedDstRow[:];                                                                                         \
        }

template< typename T, typename KER_T, typename SUM_T, SUM_T minSum, SUM_T maxSum, int cn, int windowSizePixConst, KER_T factorConst, bool isEasy, bool normalized, int xTileTSize, int yTileTSize >
__declspec(target(gfx_kernel))
void boxFilter_noBorders_tiled(const T* restrict srcptr, int srcStep,
                                T* restrict dstptr, int dstStep,
                                int xItersStart, int xItersEnd, int yItersStart, int yItersEnd,
                                KER_T factorVar, int xWindowSizePixVar, int yWindowSizePixVar) {

    cilk_for (int yTile = yItersStart; yTile <= yItersEnd; yTile++)
        cilk_for (int xTile = xItersStart; xTile <= xItersEnd; xTile++) {
            const KER_T factor = (factorConst == (KER_T) 0) ? factorVar : factorConst;
            const int xWindowSizePix = windowSizePixConst < 0 ? xWindowSizePixVar : windowSizePixConst;
            const int yWindowSizePix = windowSizePixConst < 0 ? yWindowSizePixVar : windowSizePixConst;
            const int yRadius = yWindowSizePix / 2;
            const int xRadius = xWindowSizePix / 2;

            const int xCacheTSize = xTileTSize + 2 * xRadius * cn;
            const int yCacheTSize = yTileTSize;

            const T (* src)[srcStep] = (const T (*)[]) srcptr;
            T (* dst)[dstStep] = (T (*)[]) dstptr;

            SUM_T slidingColRes[xCacheTSize];
            #pragma unroll
            slidingColRes[:] = (SUM_T)0;

            {
                unrolled_for (int y = 0; y < yWindowSizePix; y++) {
                    //T cachedSrcRow[xCacheTSize];
                    //cachedSrcRow[:] = src[yTile * yTileTSize + y - yRadius][(xTile * xTileTSize - cn * xRadius) : xCacheTSize];
                
                    //slidingColRes[0 : xTileTSize] += src[yTile * yTileTSize + y - yRadius][(xTile * xTileTSize - cn * xRadius) : xTileTSize];
                    //slidingColRes[xTileTSize : 2 * xRadius * cn] += src[yTile * yTileTSize + y - yRadius][(xTile * xTileTSize - cn * xRadius + xTileTSize) : 2 * xRadius * cn];
                    
                    #pragma unroll
                    slidingColRes[:] += src[yTile * yTileTSize + y - yRadius][(xTile * xTileTSize - cn * xRadius) : xCacheTSize];
                }

                computeWriteRow(slidingColRes, 0);
            }

            for (int y = yWindowSizePix; y < yTileTSize + yRadius * 2; y++) {
                //T cachedSrcRowPrev[xCacheTSize];
                //T cachedSrcRow[xCacheTSize];
                
                //memcpy_(cachedSrcRowPrev, &src[yTile * yTileTSize + y - yWindowSizePix - yRadius][(xTile * xTileTSize - cn * xRadius)], sizeof(cachedSrcRow));
                //memcpy_(cachedSrcRow, &src[yTile * yTileTSize + y - yRadius][(xTile * xTileTSize - cn * xRadius)], sizeof(cachedSrcRow));
                //
                //#pragma unroll
                //cachedSrcRow[:] = src[yTile * yTileTSize + y - yRadius][(xTile * xTileTSize - cn * xRadius) : xCacheTSize];
                //#pragma unroll
                //cachedSrcRowPrev[:] = src[yTile * yTileTSize + y - yWindowSizePix - yRadius][(xTile * xTileTSize - cn * xRadius) : xCacheTSize];
                
                //slidingColRes[0 : xTileTSize] += (SUM_T)src[yTile * yTileTSize + y - yRadius][(xTile * xTileTSize - cn * xRadius) : xTileTSize] -
                //                    (SUM_T)src[yTile * yTileTSize + y - yWindowSizePix - yRadius][(xTile * xTileTSize - cn * xRadius) : xTileTSize];
                //slidingColRes[xTileTSize : 2 * xRadius * cn] += (SUM_T)src[yTile * yTileTSize + y - yRadius][(xTile * xTileTSize - cn * xRadius + xTileTSize) : 2 * xRadius * cn] -
                //                    (SUM_T)src[yTile * yTileTSize + y - yWindowSizePix - yRadius][(xTile * xTileTSize - cn * xRadius + xTileTSize) : 2 * xRadius * cn];
                
                #pragma unroll
                slidingColRes[:] += (SUM_T)src[yTile * yTileTSize + y - yRadius][(xTile * xTileTSize - cn * xRadius) : xCacheTSize] -
                                    (SUM_T)src[yTile * yTileTSize + y - yWindowSizePix - yRadius][(xTile * xTileTSize - cn * xRadius) : xCacheTSize];
                
                //#pragma unroll
                //slidingColRes[:] += (SUM_T)cachedSrcRow[:] -
                //                    (SUM_T)cachedSrcRowPrev[:];

                computeWriteRow(slidingColRes, y - yWindowSizePix + 1);
            }

        }
}


struct BoxFilterGpuTask {
    bool async;

    Offload2dBuffer src;
    Offload2dBuffer dst;
    //int ddepth;
    Size ksize;

    using LayoutT = OpLayoutsServer::PrimeLayoutT;
    LayoutT layout;

    int cn;
};


class BoxFilterAdviser : public Adviser {
public:
    BoxFilterAdviser() {}

    bool accept(InputArray _src, OutputArray _dst, int ddepth,
                           Size ksize, Point anchor, bool normalize, int borderType) {
        const int xD = ksize.width;  
        const int yD = ksize.height;                                                                                                         
        const int yR = xD / 2;                                                                                                     
        const int xR = yD / 2;
        if (!inRange(xD * yD, 3, 31 * 31))
            return false;
        if (xD != yD)
            return false;

        if (_src.type() != CV_8UC3)  
            return false;
        
        const int cn = _src.channels();  

        float gpuPart = 1.0f;

        Mat src = _src.getMat();
        Mat dst = _dst.getMat();

        //if (src.total() * src.elemSize() < 60000000 / xD || src.cols < yD * 5 || src.rows * src.channels() < xD * 43 || yD % 2 != 1 || xD % 2 != 1)
        //    return false;

        const int xTileTSize = 128;
        const int yTileTSize = yD * 8;
        
        WorkerDescriptor gpu(AtomsPerIteration(xTileTSize, yTileTSize), false);
        WorkerDescriptor cpu(AtomsPerIteration(dst.elemSize(), 1), true);
        
        if (layout.accept(WorkSize(dst.elemSize() * dst.cols, dst.rows), gpuPart, gpu, cpu, cpu,
                          clarifyBordersSize(BordersSize(dst.elemSize() * anchor.x, anchor.y,
                                                         dst.elemSize() * (2 * xR - anchor.x), 2 * yR - anchor.y),
                                             src, borderType, dst.elemSize())) < 0.f)
            return false;
        
        task.layout = layout.primeLayout();
        //c_stdout << "task.layout " << task.layout.workRects()[0] << endl;

        task.async = false;
        task.ksize = ksize;
        task.cn = cn;

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
        task.src.buffer = (uchar*)task.src.buffer + dst.elemSize() * (xR - anchor.x) + task.src.stepBytes * (yR - anchor.y);

        return true;
    }

    using GpuTaskT = BoxFilterGpuTask;
    const GpuTaskT& advisedGpu() const {
        return task;
    }

private:
    BoxFilterGpuTask task;
};


class BoxFilter {
public:
    BoxFilter() {}

    int start(const BoxFilterGpuTask& _task) {
        task = &_task;
        
        const int cn = 3;

        switch(task->ksize.width) {
                case 3: return inputCase<3, 128, cn, true, true>();
                case 5: return inputCase<5, 128, cn, true, true>();
                case 7: return inputCase<7, 128, cn, true, true>();
                case 9: return inputCase<9, 128, cn, true, true>();
                case 11: return inputCase<11, 128, cn, true, true>();
                //case 13: return inputCase<13, 128, cn, true, true>();
                //case 15: return inputCase<15, 128, cn, true, true>();
                //case 17: return inputCase<17, 128, cn, true, true>();
                //case 19: return inputCase<19, 128, cn, true, true>();
                //case 21: return inputCase<21, 128, cn, true, true>();
                //case 23: return inputCase<23, 128, cn, true, true>();
                //case 25: return inputCase<25, 128, cn, true, true>();
                //case 27: return inputCase<27, 128, cn, true, true>();
                //case 29: return inputCase<29, 128, cn, true, true>();
                //case 31: return inputCase<31, 128, cn, true, true>();
        }

        throw 1;
        return 0;
    }
    
    void finalize(int taskId) {
        if (taskId)
            _GFX_wait(taskId);
        __cache += "taskId " + std::to_string(taskId) + "\n";
        //else
        //    std::cerr << "finalize taskId == 0" << std::endl;
    }
    
private:

    template<int d, int xTileTSize, int cn, bool isEasy, bool normalized>
    int inputCase() {
        Offload2dBuffer src(task->src);
        Offload2dBuffer dst(task->dst);

        Offload2dBuffer* tmp[2] = {&src, &dst};
        auto workRects = prepare(task->layout.workRects(), tmp,
                task->layout.offset());

        int lastTaskId = 0;
        for (const auto& work : workRects) {
            if (!(work.width * work.height))
                continue;
            const int yTileTSize = d * 8;
            int taskId = 1 ? 0 : _GFX_offload(&boxFilter_noBorders_tiled<uchar, int, int, 0, 255, cn, d, d * d, isEasy, normalized, xTileTSize, yTileTSize >,
                                (const uchar*)src.buffer, src.step,
                                (uchar*)dst.buffer, dst.step,                                                       
                                work.x, work.x + work.width - 1, work.y, work.y + work.height - 1,  
                                task->ksize.width * task->ksize.height, task->ksize.width, task->ksize.height);
            boxFilter_noBorders_tiled<uchar, int, int, 0, 255, cn, d, d * d, isEasy, normalized, xTileTSize, yTileTSize >(
                                (const uchar*)src.buffer, src.step,
                                (uchar*)dst.buffer, dst.step,                                                       
                                work.x, work.x + work.width - 1, work.y, work.y + work.height - 1,  
                                task->ksize.width * task->ksize.height, task->ksize.width, task->ksize.height);
            
            if (_GFX_get_last_error() != GFX_SUCCESS) {
                std::cerr << "_GFX_get_last_error() != GFX_SUCCESS" << std::endl;
                finalize(lastTaskId);
                throw 1;
            }

            lastTaskId = taskId;
        }

        return lastTaskId;
    }

    const BoxFilterGpuTask* task;
};


bool boxFilter( InputArray _src, OutputArray _dst, int ddepth,
                           Size ksize, Point anchor, bool normalize, int borderType) {
    Mat src = _src.getMat();
    _dst.create(src.size(), src.type());
    Mat dst = _dst.getMat();
    
    if( ddepth < 0 )
        ddepth = src.depth();

    //Point anchor = anchor0;//normalizeAnchor(anchor0, ksize);

    return performHeteroTask(BoxFilterAdviser(), BoxFilter(), &cv::boxFilter, src, dst, ddepth, ksize, anchor, normalize, borderType);
}

}
}
