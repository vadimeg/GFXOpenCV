#include "../../../core/src/igpu/build_list.hpp"
#ifdef BOXFILTER_BUILD

#include "../../../core/src/igpu/cv_igpu_interface.hpp"
#include "../../../core/src/igpu/igpu_comp_primitives.hpp"
#include "boxFilter.hpp"

namespace cv {

void boxFilter_cpu( InputArray _src, OutputArray _dst, int ddepth,
                Size ksize, Point anchor,
                bool normalize, int borderType );

namespace igpu {

//#define computeHardRow(source)                                                                                                              \
//            unrolled_for (int c = 0; c < cn; c++) {                                                                                                    \
//                SUM_T slidingRes = __sec_reduce_add[source[c : xWindowSizePix : cn]];                                                               \
//                                                                                                                                                \
//                cachedDstRow[c] = asResult(slidingRes);                                                                                          \
//                                                                                                                                                \
//                for (int xColRes = cn * xWindowSizePix + c; xColRes < xCacheTSize; xColRes += cn) {                                         \
//                    slidingRes += source[xColRes] - source[xColRes - cn * xWindowSizePix];                                                          \
//                    cachedDstRow[xColRes - cn * xWindowSizePix + cn] = asResult(slidingRes);                                                    \
//                }                                                                                                                               \
//            }                                     
//
//#define computeEasyRow(source)                                                                                                                    \
//            simd_for (int xColRes = cn * xRadius; xColRes < cn * xRadius + xTileTSize; xColRes++) {                                                               \
//                SUM_T res = __sec_reduce_add[source[xColRes - cn * xRadius : xWindowSizePix : cn]];                                                                   \
//                cachedDstRow[xColRes - cn * xRadius] = asResult(res);                                                                               \
//            }

#define asResult(res) ((T)(normalized ? ((res) / factor) : clamp((res) / factor, minSum, maxSum)))

#define computeHardRow(source)                                                                                                              \
            for (int c = 0; c < cn; c++) {                                                                                                    \
                SUM_T slidingRes = (SUM_T) 0;                                                                                                       \
                for (int i = 0; i < xWindowSizePix; i++)                                                                                       \
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
                SUM_T res = (SUM_T) 0;                                                          \
                __pragma(static_max(xWindowSize, 5)) for (int xColSum = -cn * (xRadius); xColSum <= cn * xRadius; xColSum += cn)                                                           \
                    res += (SUM_T) source[xColRes + xColSum];                                                                                                      \
                cachedDstRow[xColRes - cn * xRadius] = asResult(res);                                                                               \
            }

#define computeWriteRow(source, yRes) {                                                                                                                                 \
            T cachedDstRow[xCacheTSize];                                                                                                              \
            if (isEasy)                                                                                                                       \
                computeEasyRow(source)                                                                                                             \
            else                                                                                                                                          \
                computeHardRow(source)                                       \
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
            slidingColRes[:] = (SUM_T)0;

            {
                //#pragma unroll(yWindowSizePix == 7 ? 4 : static_min(yWindowSizePix, 5))
                //#pragma unroll
                //#pragma unroll(static_min(yWindowSizePix, 7))
                for (int y = 0; y < yWindowSizePix; y++) {
                    //T cachedSrcRow[xCacheTSize];
                    //cachedSrcRow[:] = src[yTile * yTileTSize + y - yRadius][(xTile * xTileTSize - cn * xRadius) : xCacheTSize];
                
                    //slidingColRes[0 : xTileTSize] += src[yTile * yTileTSize + y - yRadius][(xTile * xTileTSize - cn * xRadius) : xTileTSize];
                    //slidingColRes[xTileTSize : 2 * xRadius * cn] += src[yTile * yTileTSize + y - yRadius][(xTile * xTileTSize - cn * xRadius + xTileTSize) : 2 * xRadius * cn];
                    
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
    LayoutT cpuLayout;
    LayoutT gpuLayout;
    
    bool normalized;

    int cn;
};


#define yTileSizes(r)           \
          (((r) == 1) ? 38 :     \
           ((r) == 2) ? 3 :       \
           ((r) == 3) ? 39 :       \
           ((r) == 4) ? 57 :       \
           ((r) == 5) ? 77 :       \
           ((r) == 6) ? 85 :       \
           ((r) == 7) ? 44 :       \
           ((r) == 8) ? (r) * 8 :       \
           ((r) == 9) ? (r) * 8 :       \
           ((r) == 10) ? (r) * 8 :       \
           ((r) == 11) ? (r) * 8 :       \
           ((r) == 12) ? (r) * 8 :       \
           ((r) == 13) ? (r) * 8 :       \
           ((r) == 14) ? (r) * 8 :       \
           ((r) == 15) ? (r) * 8 : -1)


class BoxFilterAdviser : public Adviser {
public:
    BoxFilterAdviser() {}

    bool accept(InputArray _src, OutputArray _dst, int ddepth,
                           Size ksize, Point anchor, bool normalize, int borderType) {
        const int xD = ksize.width;  
        const int yD = ksize.height;                                                                                                         
        const int yR = xD / 2;                                                                                                     
        const int xR = yD / 2;

        if (borderType & BORDER_ISOLATED)
            return false;
        
        if (!inRange(xD * yD, 3 * 3, 31 * 31))
            return false;
        if (xD % 2 != 1)
            return false;
        if (yD % 2 != 1)
            return false;
        if (xD != yD)
            return false;

        if (_src.type() != CV_8UC3)  
            return false;
        
        const int cn = _src.channels();  

        float gpuPart = getGpuPart();

        try {

            src = _src.getMat();
            Mat dst = _dst.getMat();
            if (src.datastart == dst.datastart)
                src = src.clone();
        
            //if (src.total() * src.elemSize() < 10000000 / xD)
            //    return false;

            task.async = false;
            task.ksize = ksize;
            task.cn = cn;
            task.normalized = normalize;

            const int xTileTSize = 128;
            const int yTileTSize = yTileSizes(xR);
        
            WorkerDescriptor gpu(AtomsPerIteration(dst.elemSize1() * xTileTSize, yTileTSize), false);
            WorkerDescriptor cpu(AtomsPerIteration(dst.elemSize(), 1), true);
        
            if (!layout.accept(WorkSize(dst.elemSize() * dst.cols, dst.rows), gpuPart, gpu, cpu, cpu,
                              clarifyBordersSize(BordersSize(dst.elemSize() * anchor.x, anchor.y,
                                                             dst.elemSize() * (2 * xR - anchor.x), 2 * yR - anchor.y),
                                                 src, borderType)))
                return false;
            
            task.gpuLayout = layout.primeLayout();

            //if (task.gpuLayout.total() < 86)
            //    return false;

            const int buffersNum = 2;
            Offload2dBuffer* buffers[buffersNum] = {&task.src, &task.dst};
            Mat* mats[buffersNum] = {&src, &dst};

            for (int i = 0; i < buffersNum; i++) {
                *buffers[i] = Offload2dBuffer(*mats[i]);
                if (!buffers[i]->share())
                    return false;
            }

            task.src.buffer = (uchar*)task.src.buffer + dst.elemSize() * (xR - anchor.x) + task.src.stepBytes * (yR - anchor.y);

            ////c_stdout << "task.layout " << task.layout.workRects()[0] << endl;

        } catch (std::exception& e) {
            return false;
        }

        return true;
    }

    using GpuTaskT = BoxFilterGpuTask;
    const GpuTaskT& advisedGpu() const {
        return task;
    }

private:
    Mat src;
    BoxFilterGpuTask task;
};


class BoxFilterIgpuWorker {
public:
    BoxFilterIgpuWorker() {}

    void start(const BoxFilterGpuTask& _task) throw (std::runtime_error)  {
        task = &_task;
        
        switch(task->ksize.width) {
                case 3: return start_withCnAndNormalizedFlag<3, true>();
                case 5: return start_withCnAndNormalizedFlag<5, true>();
                case 7: return start_withCnAndNormalizedFlag<7, true>();
                case 9: return start_withCnAndNormalizedFlag<9, true>();
                case 11: return start_withCnAndNormalizedFlag<11, true>();
                case 13: return start_withCnAndNormalizedFlag<13, true>();
                case 15: return start_withCnAndNormalizedFlag<15, true>();
                case 17: return start_withCnAndNormalizedFlag<17, true>();
                case 19: return start_withCnAndNormalizedFlag<19, true>();
                case 21: return start_withCnAndNormalizedFlag<21, true>();
                case 23: return start_withCnAndNormalizedFlag<23, true>();
                case 25: return start_withCnAndNormalizedFlag<25, true>();
                case 27: return start_withCnAndNormalizedFlag<27, true>();
                case 29: return start_withCnAndNormalizedFlag<29, true>();
                case 31: return start_withCnAndNormalizedFlag<31, true>();
        }

		throw std::runtime_error("BoxFilterIgpuWorker start: unsupported params");
    }
    
    void finalize() {
        if (lastTaskId && _GFX_wait(lastTaskId))
		    throw std::runtime_error("BoxFilterIgpuWorker finalize: _GFX_get_last_error() != GFX_SUCCESS");
        lastTaskId = 0;
        //c_stdout << "finalize taskId " << lastTaskId << std::endl;
    }
    
private:
	GfxTaskId lastTaskId = 0;

    template<int d, bool isEasy>
    void start_withCnAndNormalizedFlag() throw (std::runtime_error) {
        if (task->cn == 3) {
            if (task->normalized) {
                startGpu<d, 128, 3, isEasy, true>();
                //startCpu<d, 128, 3, isEasy, true>();
                return;
            } else {
                startGpu<d, 128, 3, isEasy, false>();
                //startCpu<d, 128, 3, isEasy, false>();
                return;
            }
        }
		throw std::runtime_error("BoxFilterIgpuWorker start: unsupported cn");
    }

    template<int d, int xTileTSize, int cn, bool isEasy, bool normalized>
    void startCpu() throw (std::runtime_error) {
        Offload2dBuffer src(task->src);
        Offload2dBuffer dst(task->dst);

        shiftWithOffset(task->cpuLayout.offset(), src, dst);
        
        auto workRects = task->cpuLayout.workRects();
        for (const auto& work : workRects) {
            if (!(work.width * work.height))
                continue;
            const int yTileTSize = yTileSizes(d / 2);
            boxFilter_noBorders_tiled<uchar, int, int, 0, 255, cn, d, d * d, isEasy, normalized, xTileTSize, yTileTSize >(
                                (const uchar*)src.buffer, src.step,
                                (uchar*)dst.buffer, dst.step,                                                       
                                work.x, work.x + work.width - 1, work.y, work.y + work.height - 1,  
                                task->ksize.width * task->ksize.height, task->ksize.width, task->ksize.height);
        }
    }

    template<int d, int xTileTSize, int cn, bool isEasy, bool normalized>
    void startGpu() throw (std::runtime_error) {
        Offload2dBuffer src(task->src);
        Offload2dBuffer dst(task->dst);

        shiftWithOffset(task->gpuLayout.offset(), src, dst);
        
        auto workRects = task->gpuLayout.workRects();
        for (const auto& work : workRects) {
            if (!(work.width * work.height))
                continue;
            const int yTileTSize = yTileSizes(d / 2);
            GfxTaskId taskId = _GFX_offload(&boxFilter_noBorders_tiled<uchar, int, int, 0, 255, cn, d, d * d, isEasy, normalized, xTileTSize, yTileTSize >,
                                (const uchar*)src.buffer, src.step,
                                (uchar*)dst.buffer, dst.step,                                                       
                                work.x, work.x + work.width - 1, work.y, work.y + work.height - 1,  
                                task->ksize.width * task->ksize.height, task->ksize.width, task->ksize.height);

            //c_stdout << "taskId " << taskId << endl;

            if (_GFX_get_last_error() != GFX_SUCCESS) {
                finalize();
                throw std::runtime_error("BoxFilterIgpuWorker start: _GFX_get_last_error() != GFX_SUCCESS");
            }

            lastTaskId = taskId;
        }
    }

    const BoxFilterGpuTask* task;
};



bool boxFilter( InputArray _src, OutputArray _dst, int ddepth,
                           Size ksize, Point anchor0, bool normalize, int borderType) {
    Mat src = _src.getMat();
    
    if( ddepth < 0 )
        ddepth = src.depth();

    if (ddepth != src.depth())
        return false;
    
    _dst.create(src.size(), CV_MAKETYPE(ddepth, src.channels()));
    Mat dst = _dst.getMat();

    Point anchor = normalizeAnchor(anchor0, ksize);

    return performHeteroTaskWithCvFunc(BoxFilterAdviser(), BoxFilterIgpuWorker(), &cv::boxFilter_cpu, src, dst, ddepth, ksize, anchor, normalize, borderType);
}
}
}

#endif