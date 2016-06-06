#include "../../../core/src/igpu/build_list.hpp"
#ifdef MINMAX_BUILD
#include "../../../core/src/igpu/cv_igpu_interface.hpp"
#include "../../../core/src/igpu/igpu_comp_primitives.hpp"
#include "minmax.hpp"

namespace cv {

void morphOp_cpu( int op, InputArray _src, OutputArray _dst, InputArray _kernel,
                          Point anchor, int iterations,
                          int borderType,
                          const Scalar& borderValue );

namespace igpu {

void morphOp_cpu_( InputArray _src, OutputArray _dst, int op, InputArray _kernel,
                          Point anchor, int iterations,
                          int borderType,
                          const Scalar& borderValue ) {
    morphOp_cpu(op, _src, _dst, _kernel, anchor, iterations, borderType, borderValue);
}

template< typename T, int cn, bool isMax, int xWindowPixSize, int yWindowPixSize, int xTileTSize, int yTileTSize >
__declspec(target(gfx_kernel))
void minmax3_noBorders_tiled(const T* restrict srcptr, int srcStep,
                                T* restrict dstptr, int dstStep,
                                int xItersStart, int xItersEnd, int yItersStart, int yItersEnd) {
    cilk_for (int yTile = yItersStart; yTile <= yItersEnd; yTile++)
        cilk_for (int xTile = xItersStart; xTile <= xItersEnd; xTile++) {
            const int xRadius = xWindowPixSize / 2;
            const int yRadius = yWindowPixSize / 2;
            const int xCacheTSize = xTileTSize + 2 * cn * xRadius;
            const int yCacheTSize = yTileTSize + 2 * yRadius;

            const T (* src)[srcStep] = (const T (*)[]) srcptr;
            T (* dst)[dstStep] = (T (*)[]) dstptr;
            
            T dstRowCachedColRes[xCacheTSize];
            
            for (int y = 0; y < yTileTSize; y++) {
                {
                    const T* src_ = &src[yTile * yTileTSize - yRadius + y][0];
                    __assume_aligned(src_, 32);
                    dstRowCachedColRes[:] = src_[xTile * xTileTSize - cn * xRadius : xCacheTSize];
                    unrolled_for (int yRes = -yRadius + 1; yRes <= yRadius; yRes++) {
                        const T* src_ = &src[yTile * yTileTSize + yRes + y][0];
                        dstRowCachedColRes[:] = isMax ?
                            max(dstRowCachedColRes[:], src_[xTile * xTileTSize - cn * xRadius : xCacheTSize]) :
                            min(dstRowCachedColRes[:], src_[xTile * xTileTSize - cn * xRadius : xCacheTSize]);
                    }

                }

                T dstRowCache[xTileTSize];
                simd_for (int x = cn * xRadius; x < cn * xRadius + xTileTSize; x++) {
                    T res = dstRowCachedColRes[x - cn * xRadius];
                    unrolled_for (int xRes = -xRadius + 1; xRes <= xRadius; xRes++)
                        res = isMax ?
                            max(res, dstRowCachedColRes[x + xRes * cn]) :
                            min(res, dstRowCachedColRes[x + xRes * cn]);
                    dstRowCache[x - cn * xRadius] = res;
                }

                dst[yTile * yTileTSize + y][xTile * xTileTSize : xTileTSize] = dstRowCache[:];
            }
        }
}


template< typename T, int cn, bool isMax, int xWindowPixSize, int yWindowPixSize, int xTileTSize, int yTileTSize >
__declspec(target(gfx_kernel))
void minmax_noBorders_tiled(const T* restrict srcptr, int srcStep,
                                T* restrict dstptr, int dstStep,
                                int xItersStart, int xItersEnd, int yItersStart, int yItersEnd) {
    cilk_for (int yTile = yItersStart; yTile <= yItersEnd; yTile++)
        cilk_for (int xTile = xItersStart; xTile <= xItersEnd; xTile++) {
            if ((xWindowPixSize * yWindowPixSize) <= 9) {
                const int xRadius = xWindowPixSize / 2;
                const int yRadius = yWindowPixSize / 2;
                const int xCacheTSize = xTileTSize + 2 * cn * xRadius;
                const int yCacheTSize = yTileTSize + 2 * yRadius;

                const T (* src)[srcStep] = (const T (*)[]) srcptr;
                T (* dst)[dstStep] = (T (*)[]) dstptr;
            
                T dstRowCachedColRes[xCacheTSize];
            
                for (int y = 0; y < yTileTSize; y++) {
                    {
                        const T* src_ = &src[yTile * yTileTSize - yRadius + y][0];
                        __assume_aligned(src_, 32);
                        dstRowCachedColRes[:] = src_[xTile * xTileTSize - cn * xRadius : xCacheTSize];
                        unrolled_for (int yRes = -yRadius + 1; yRes <= yRadius; yRes++) {
                            const T* src_ = &src[yTile * yTileTSize + yRes + y][0];
                            dstRowCachedColRes[:] = isMax ?
                                max(dstRowCachedColRes[:], src_[xTile * xTileTSize - cn * xRadius : xCacheTSize]) :
                                min(dstRowCachedColRes[:], src_[xTile * xTileTSize - cn * xRadius : xCacheTSize]);
                        }

                    }

                    T dstRowCache[xTileTSize];
                    simd_for (int x = cn * xRadius; x < cn * xRadius + xTileTSize; x++) {
                        T res = dstRowCachedColRes[x - cn * xRadius];
                        unrolled_for (int xRes = -xRadius + 1; xRes <= xRadius; xRes++)
                            res = isMax ?
                                max(res, dstRowCachedColRes[x + xRes * cn]) :
                                min(res, dstRowCachedColRes[x + xRes * cn]);
                        dstRowCache[x - cn * xRadius] = res;
                    }

                    dst[yTile * yTileTSize + y][xTile * xTileTSize : xTileTSize] = dstRowCache[:];
                }
            } else {
                const int xRadius = xWindowPixSize / 2;
                const int yRadius = yWindowPixSize / 2;
                const int xCacheTSize = xTileTSize + 2 * cn * xRadius;
                const int yCacheTSize = yTileTSize + 2 * yRadius;

                const T (* src)[srcStep] = (const T (*)[]) srcptr;
                T (* dst)[dstStep] = (T (*)[]) dstptr;
            
                T dstRowCachedColRes[yTileTSize][xCacheTSize];

                T dstRowCachedSubRes[xCacheTSize];

                //for (int y = 0; y < 2 * yRadius; y++) {
                //    T dstRowCachedSrcRow[xCacheTSize];
                //    
                //    const T* src_ = &src[yTile * yTileTSize - yRadius + y][0];
                //    __assume_aligned(src_, 32);
                //    dstRowCachedSrcRow[:] = src_[xTile * xTileTSize - cn * xRadius : xCacheTSize];
                //    
                //    if ((y % yWindowPixSize) == 0)
                //        dstRowCachedSubRes[:] = dstRowCachedSrcRow[0 : xCacheTSize];
                //    else
                //        dstRowCachedSubRes[:] = min(dstRowCachedSubRes[:], dstRowCachedSrcRow[0 : xCacheTSize]);
                //}
            
                for (int y = 0; y < yCacheTSize - 0; y++) {
                    T dstRowCachedSrcRow[xCacheTSize];
                
                    const T* src_ = &src[yTile * yTileTSize - yRadius + y][0];
                    __assume_aligned(src_, 32);
                    dstRowCachedSrcRow[:] = src_[xTile * xTileTSize - cn * xRadius : xCacheTSize];
                
                    if ((y % yWindowPixSize) == 0)
                        dstRowCachedSubRes[:] = dstRowCachedSrcRow[0 : xCacheTSize];
                    else
                        dstRowCachedSubRes[:] = isMax ? 
                            max(dstRowCachedSubRes[:], dstRowCachedSrcRow[0 : xCacheTSize]) :
                            min(dstRowCachedSubRes[:], dstRowCachedSrcRow[0 : xCacheTSize]);
            
                    if (y >= 2 * yRadius)
                    dstRowCachedColRes[y - 2 * yRadius][:] = dstRowCachedSubRes[:];
                }
            
                dstRowCachedSubRes[:] = src[yTile * yTileTSize - yRadius + yCacheTSize - 1][xTile * xTileTSize - cn * xRadius : xCacheTSize];
            
                //for (int y = yCacheTSize - 2; y >= yCacheTSize - 2 * yRadius; y++) {
                //    T dstRowCachedSrcRow[xCacheTSize];
                //    
                //    if ((y % yWindowPixSize) == 0)
                //        continue;
                //    
                //    const T* src_ = &src[yTile * yTileTSize - yRadius + y][0];
                //    __assume_aligned(src_, 32);
                //    dstRowCachedSrcRow[:] = src_[xTile * xTileTSize - cn * xRadius : xCacheTSize];
                //    
                //    if ((y % yWindowPixSize) == (yWindowPixSize - 1))
                //        dstRowCachedSubRes[:] = dstRowCachedSrcRow[:];
                //    else
                //        dstRowCachedSubRes[:] = min(dstRowCachedSubRes[:], dstRowCachedSrcRow[:]);
                //}

                for (int y = yCacheTSize - 2; y >= 0; y--) {
                    T dstRowCachedSrcRow[xCacheTSize];
                
                    if ((y % yWindowPixSize) == 0)
                        continue;
                
                    const T* src_ = &src[yTile * yTileTSize - yRadius + y][0];
                    __assume_aligned(src_, 32);
                    dstRowCachedSrcRow[:] = src_[xTile * xTileTSize - cn * xRadius : xCacheTSize];
                
                    if ((y % yWindowPixSize) == (yWindowPixSize - 1))
                        dstRowCachedSubRes[:] = dstRowCachedSrcRow[:];
                    else
                        dstRowCachedSubRes[:] = isMax ? 
                            max(dstRowCachedSubRes[:], dstRowCachedSrcRow[:]) :
                            min(dstRowCachedSubRes[:], dstRowCachedSrcRow[:]);
            
                    if (y < yTileTSize)
                        dstRowCachedColRes[y][:] = isMax ? 
                            max(dstRowCachedColRes[y][:], dstRowCachedSubRes[:]) :
                            min(dstRowCachedColRes[y][:], dstRowCachedSubRes[:]);
                }

                //for (int y = 0; y < yCacheTSize - 0; y++) {
                //    T dstRowCachedSrcRow[xCacheTSize];
                //    
                //    const T* src_ = &src[yTile * yTileTSize - yRadius + y][0];
                //    __assume_aligned(src_, 32);
                //    dstRowCachedSrcRow[:] = src_[xTile * xTileTSize - cn * xRadius : xCacheTSize];
                //    
                //    if ((y % yWindowPixSize) == 0)
                //        dstRowCachedSubRes[:] = dstRowCachedSrcRow[0 : xCacheTSize];
                //    else
                //        dstRowCachedSubRes[:] = min(dstRowCachedSubRes[y - 1][:], dstRowCachedSrcRow[0 : xCacheTSize]);
                //}
                //
                //dstRowCachedSubRes[yCacheTSize - 1][:] = src[yTile * yTileTSize - yRadius + yCacheTSize - 1][xTile * xTileTSize - cn * xRadius : xCacheTSize];
                //for (int y = yCacheTSize - 2; y >= 0; y--) {
                //    T dstRowCachedSrcRow[xCacheTSize];
                //
                //    if ((y % yWindowPixSize) == 0) {
                //        dstRowCachedSubRes[:] = (T) 0;
                //        continue;
                //    }
                //
                //    const T* src_ = &src[yTile * yTileTSize - yRadius + y][0];
                //    __assume_aligned(src_, 32);
                //    dstRowCachedSrcRow[:] = src_[xTile * xTileTSize - cn * xRadius : xCacheTSize];
                //    
                //    if ((y % yWindowPixSize) == (yWindowPixSize - 1))
                //        dstRowCachedSubRes[:] = dstRowCachedSrcRow[:];
                //    else
                //        dstRowCachedSubRes[:] = min(dstRowCachedSubRes[y + 1][:], dstRowCachedSrcRow[:]);
                //}

                for (int y = 0; y < yTileTSize; y++) { //naive due to register file size constraints
                    T dstRowCache[xTileTSize];
                    simd_for (int x = cn * xRadius; x < cn * xRadius + xTileTSize; x++) {
                        T res = dstRowCachedColRes[y][x - cn * xRadius];
                        for (int xRes = -xRadius + 1; xRes <= xRadius; xRes++)
                            res = isMax ?
                                max(res, dstRowCachedColRes[y][x + xRes * cn]) :
                                min(res, dstRowCachedColRes[y][x + xRes * cn]);
                        dstRowCache[x - cn * xRadius] = res;
                    }

                    dst[yTile * yTileTSize + y][xTile * xTileTSize : xTileTSize] = dstRowCache[:];
                }
            }
        }
}


struct MinMaxGpuTask {
    bool async;

    Offload2dBuffer src;
    Offload2dBuffer dst;

    int d;

    using LayoutT = OpLayoutsServer::PrimeLayoutT;
    LayoutT cpuLayout;
    LayoutT gpuLayout;

    bool isMax;

    int cn;
};

#define yTileSizes(r)           \
          (((r) == 1 ) ? (r) * 4 :     \
           ((r) == 2 ) ? (r) * 4 :       \
           ((r) == 3 ) ? (r) * 4:       \
           ((r) == 4 ) ? (r) * 4:       \
           ((r) == 5 ) ? (r) * 4 :       \
           ((r) == 6 ) ? (r) * 4:       \
           ((r) == 7 ) ? (r) * 4:       \
           ((r) == 8 ) ? (r) * 4 :       \
           ((r) == 9 ) ? (r) * 4:       \
           ((r) == 10) ? (r) * 4:       \
           ((r) == 11) ? (r) * 4 :       \
           ((r) == 12) ? (r) * 4:       \
           ((r) == 13) ? (r) * 4:       \
           ((r) == 14) ? (r) * 4 :       \
           ((r) == 15) ? (r) * 4:       \
           ((r) == 16) ? (r) * 4:       \
           ((r) == 17) ? (r) * 4:       \
           ((r) == 18) ? (r) * 4:       \
           ((r) == 19) ? (r) * 4:       \
           ((r) == 20) ? (r) * 4:       \
           ((r) == 21) ? (r) * 4:       \
           ((r) == 22) ? (r) * 4:       \
           ((r) == 23) ? (r) * 4: 0)

class MinMaxAdviser : public Adviser {
public:
    MinMaxAdviser(bool isMax_) : isMax(isMax_) {}

    bool accept(InputArray _src, OutputArray _dst, int unused, InputArray kernel,
                          Point anchor = Point(-1,-1), int iterations = 1,
                          int borderType = BORDER_CONSTANT,
                          const Scalar& borderValue = morphologyDefaultBorderValue()) {
        const int d = kernel.cols();                                                                                                          
        const int r = d / 2;

        if (kernel.cols() != kernel.rows())
            return false;

        if (iterations != 1)
            return false;

        Mat kmat = kernel.getMat();
        if (countNonZero(kmat) != kernel.rows() * kernel.cols())
            return false;

        if (borderType & BORDER_ISOLATED)
            return false;
        
        if (!inRange(d, 3, 39))
            return false;
        if (d % 2 != 1)
            return false;

        if (_src.type() != CV_8UC3)  
            return false;
        
        const int cn = _src.channels();  

        float gpuPart = getGpuPart();

        try {

            Mat src = _src.getMat();
            Mat dst = _dst.getMat();
            if (src.datastart == dst.datastart)
                return false;
        
            //if (src.total() * src.elemSize() < 10000000 / xD)
            //    return false;

            task.async = false;
            task.cn = cn;
            task.d = d;
            task.isMax = isMax;

            const int yTileTSize = yTileSizes(r);
        
            WorkerDescriptor gpu(AtomsPerIteration(dst.elemSize1() * 128, yTileTSize), false);
            WorkerDescriptor cpu(AtomsPerIteration(dst.elemSize1() * 256, yTileTSize), false);
            WorkerDescriptor border(AtomsPerIteration(dst.elemSize(), 1), true);
        
            if (!layout.accept(WorkSize(dst.elemSize() * dst.cols, dst.rows), gpuPart, gpu, cpu, border,
                              clarifyBordersSize(BordersSize(dst.elemSize() * anchor.x, anchor.y,
                                                             dst.elemSize() * (2 * r - anchor.x), 2 * r - anchor.y),
                                                 src, borderType)))
                return false;
            
            task.gpuLayout = layout.primeLayout();
            task.cpuLayout = layout.secondLayout();

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

            task.src.buffer = (uchar*)task.src.buffer + dst.elemSize() * (r - anchor.x) + task.src.stepBytes * (r - anchor.y);

            //c_stdout << "task.gpuLayout " << task.gpuLayout.workRects()[0] << endl;
            //c_stdout << "task.cpuLayout " << task.cpuLayout.workRects()[0] << endl;


        } catch (std::exception& e) {
            return false;
        }

        return true;
    }

    using GpuTaskT = MinMaxGpuTask;
    const GpuTaskT& advisedGpu() const {
        return task;
    }

private:
    bool isMax;
    MinMaxGpuTask task;
};


class MinMaxIgpuWorker {
public:
    MinMaxIgpuWorker() {}

    void start(const MinMaxGpuTask& _task) throw (std::runtime_error)  {
        task = &_task;
        
        switch(task->d) {
            case 3: return start_withCn<3>();
            case 5: return start_withCn<5>();
            case 7: return start_withCn<7>();
            case 9: return start_withCn<9>();
            case 11: return start_withCn<11>();
            case 13: return start_withCn<13>();
            case 15: return start_withCn<15>();
            case 17: return start_withCn<17>();
            case 19: return start_withCn<19>();
            case 21: return start_withCn<21>();
            case 23: return start_withCn<23>();
            case 25: return start_withCn<25>();
            case 27: return start_withCn<27>();
            case 29: return start_withCn<29>();
            case 31: return start_withCn<31>();
            case 33: return start_withCn<33>();
            case 35: return start_withCn<35>();
            case 37: return start_withCn<37>();
            case 39: return start_withCn<39>();
        }

		throw std::runtime_error("MinMaxIgpuWorker start: unsupported params");
    }
    
    void finalize() {
        if (lastTaskId && _GFX_wait(lastTaskId))
		    throw std::runtime_error("MinMaxIgpuWorker finalize: _GFX_get_last_error() != GFX_SUCCESS");

        lastTaskId = 0;
        ////c_stdout << "finalize taskId " << lastTaskId << std::endl;
    }
    
private:
	GfxTaskId lastTaskId = 0;

    template<int d>
    void start_withCn() throw (std::runtime_error) {
        if (task->cn == 3) {
            if (task->isMax) {
                startGpu<d, 128, 3, true>();
                startCpu<d, 256, 3, true>();
                return;
            } else {
                startGpu<d, 128, 3, false>();
                startCpu<d, 256, 3, false>();
                return;
            }
        }
		throw std::runtime_error("MinMaxIgpuWorker start: unsupported cn");
    }

    template<int d, int xTileTSize, int cn, bool isMax>
    void startCpu() throw (std::runtime_error) {
        Offload2dBuffer src(task->src);
        Offload2dBuffer dst(task->dst);

        shiftWithOffset(task->cpuLayout.offset(), src, dst);
        
        auto workRects = task->cpuLayout.workRects();
        for (const auto& work : workRects) {
            //c_stdout << "work " << work << endl;
            if (!(work.width * work.height))
                continue;
            const int yTileTSize = yTileSizes(d / 2);
            minmax_noBorders_tiled<uchar, cn, isMax, d, d, xTileTSize, yTileTSize >(
                                    (const uchar*)src.buffer, src.step,
                                    (uchar*)dst.buffer, dst.step,                                                       
                                    work.x, work.x + work.width - 1, work.y, work.y + work.height - 1);
        }
    }
    
    template<int d, int xTileTSize, int cn, bool isMax>
    void startGpu() throw (std::runtime_error) {
        Offload2dBuffer src(task->src);
        Offload2dBuffer dst(task->dst);

        shiftWithOffset(task->gpuLayout.offset(), src, dst);
        
        auto workRects = task->gpuLayout.workRects();
        for (const auto& work : workRects) {
            if (!(work.width * work.height))
                continue;
            const int yTileTSize = yTileSizes(d / 2);

            GfxTaskId taskId = _GFX_offload(&minmax_noBorders_tiled<uchar, cn, isMax, d, d, xTileTSize, yTileTSize >,
                                    (const uchar*)src.buffer, src.step,
                                    (uchar*)dst.buffer, dst.step,                                                       
                                    work.x, work.x + work.width - 1, work.y, work.y + work.height - 1);

            //c_stdout << "taskId " << taskId << endl;

            if (_GFX_get_last_error() != GFX_SUCCESS) {
                finalize();
                throw std::runtime_error("MinMaxIgpuWorker start: _GFX_get_last_error() != GFX_SUCCESS");
            }

            lastTaskId = taskId;
        }
    }

    const MinMaxGpuTask* task;
};



bool morphOp( int op, InputArray _src, OutputArray _dst, InputArray _kernel,
                          Point anchor, int iterations,
                          int borderType,
                          const Scalar& borderValue ) {

    Mat kernel = _kernel.getMat();
    Size ksize = kernel.size();

    if (iterations == 0 || kernel.total() == 1)
        return false;

    if (kernel.empty()) {
        kernel = getStructuringElement(MORPH_RECT, Size(1 + iterations * 2, 1 + iterations * 2));
        anchor = Point(iterations, iterations);
        iterations = 1;
    } else if(iterations > 1 && countNonZero(kernel) == kernel.rows * kernel.cols) {
        anchor = Point(anchor.x * iterations, anchor.y * iterations);
        kernel = getStructuringElement(MORPH_RECT,
                                       Size(ksize.width + (iterations - 1) * (ksize.width - 1),
                                            ksize.height + (iterations - 1) * (ksize.height - 1)),
                                       anchor);
        iterations = 1;
    }

    Mat src = _src.getMat();
    
    _dst.create(src.size(),src.type());
    Mat dst = _dst.getMat();

    anchor = normalizeAnchor(anchor, kernel.size());

    if (iterations < 0)
        iterations = 1;

    if (op == MORPH_DILATE) {
        while(iterations--)
            if (!performHeteroTaskWithCvFunc_onlyBorders(MinMaxAdviser(true), MinMaxIgpuWorker(), &morphOp_cpu_, src, dst, op, kernel, anchor, 1, borderType, borderValue))
                return false;
        return true;
    } else if (op == MORPH_ERODE) {
        while(iterations--)
            if (!performHeteroTaskWithCvFunc_onlyBorders(MinMaxAdviser(false), MinMaxIgpuWorker(), &morphOp_cpu_, src, dst, op, kernel, anchor, 1, borderType, borderValue))
                return false;
        return true;
    }
    return false;
}


}
}

#endif