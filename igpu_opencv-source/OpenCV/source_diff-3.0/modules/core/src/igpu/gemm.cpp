#include "../../../core/src/igpu/build_list.hpp"
#ifdef GEMM_BUILD
#include "cv_igpu_interface.hpp"
#include "igpu_comp_primitives.hpp"
#include "gemm.hpp"

__declspec(target(gfx_kernel))
void tmpGemm(int* ptr) {
    cilk_for (int yTile = 0; yTile <= 0; yTile++)
        *ptr = 0;
}

namespace cv {

void gemm_cpu( InputArray matA, InputArray matB, double alpha,
           InputArray matC, double beta, OutputArray _matD, int flags );

namespace igpu {


template< typename AT, typename BT, typename DST_T, typename SUM_T, int cn, int xTileTSize, int yTileTSize >
__declspec(target(gfx_kernel))
void matmult_tiled(const AT* restrict aptr, int aStep,
                                const BT* restrict bptr, int bStep,
                                DST_T* restrict dstptr, int dstStep,
                                int xItersStart, int xItersEnd, int yItersStart, int yItersEnd,
                                int srcTiles, SUM_T alpha) {

    /*                    A                             B (transposed)
                   ---------------                     ---------------
      yTileTSize   ***|***|***|***        xTileTSize   ***|***|***|***
                   ***|***|***|***                     ***|***|***|***
                   ---------------                     ***|***|***|***
                   ---------------                     ---------------
                   xTileTSize * n                      xTileTSize * n
    */

    cilk_for (int yTile = yItersStart; yTile <= yItersEnd; yTile++)
        cilk_for (int xTile = xItersStart; xTile <= xItersEnd; xTile++) {
#define asResult(res) ((DST_T)(alpha * (res)))
//#define asResult(res) res

            const AT (* a)[aStep] = (const AT (*)[]) aptr;
            const BT (* b)[bStep] = (const BT (*)[]) bptr;
            DST_T (* dst)[dstStep] = (DST_T (*)[]) dstptr;
            
            DST_T dstCached[yTileTSize][xTileTSize];
            dstCached[:][:] = (DST_T) 0;
            
            for (int srcTile = 0; srcTile < srcTiles; srcTile++) {
                AT aCached[yTileTSize][xTileTSize];
                BT bCached[xTileTSize];
            
                aCached[:][:] = a[yTile * yTileTSize : yTileTSize][srcTile * xTileTSize : xTileTSize];
                
                unrolled_for (int yBLocal = 0; yBLocal < xTileTSize; yBLocal++) {
                    BT bCached[xTileTSize];
                    bCached[:] = b[srcTile * xTileTSize + yBLocal][xTile * xTileTSize : xTileTSize];
                    
                    unrolled_for (int yALocal = 0; yALocal < yTileTSize; yALocal++)
                        if (cn == 1)
                            dstCached[yALocal][:] += aCached[yALocal][yBLocal] * bCached[:];
                        //else if (cn == 2) {
                        //    dstCached[yALocal][0 : xTileTSize / 2 : 2] +=
                        //        aCached[yALocal][0 : xTileTSize / 2 : 2] * bCached[0 : xTileTSize / 2 : 2] -
                        //        aCached[yALocal][1 : xTileTSize / 2 : 2] * bCached[1 : xTileTSize / 2 : 2];
                        //    dstCached[yALocal][1 : xTileTSize / 2 : 2] +=
                        //        aCached[yALocal][0 : xTileTSize / 2 : 2] * bCached[1 : xTileTSize / 2 : 2] -
                        //        aCached[yALocal][1 : xTileTSize / 2 : 2] * bCached[0 : xTileTSize / 2 : 2];
                        //}
                }
            }
            
            dst[yTile * yTileTSize : yTileTSize][xTile * xTileTSize : xTileTSize] = asResult(dstCached[:][:]);
        }
}


struct MatmultGpuTask {
    bool async;

    Offload2dBuffer src1;
    Offload2dBuffer src2;
    Offload2dBuffer dst;

    int srcTiles;

    float alpha;

    int cn;
    
    using LayoutT = OpLayoutsServer::PrimeLayoutT;
    LayoutT layout;
};


class GemmAdviser : public Adviser {
public:
    GemmAdviser() {}

    bool accept(InputArray _src1, InputArray _src2, double alpha,
                       InputArray _src3, double beta, OutputArray _dst, int flags) {
            //c_stdout << "accept " << endl;
        if (_src1.cols() != _src2.rows())
            return false;
        //if (_src1.total() < 1)
        //    return false;

        if (_src1.type() != CV_32FC1)  
            return false;
        if (_src2.type() != CV_32FC1)  
            return false;

        try {

            Mat src1 = _src1.getMat();
            Mat src2 = _src2.getMat();
            Mat dst = _dst.getMat();

            if (src1.cols % 16 || src1.rows % 16 || src2.cols % 16 || src2.rows % 16)
                return false;
        
            const int xTileTSize = 16;
            const int yTileTSize = 16;
        
            WorkerDescriptor gpu(AtomsPerIteration(dst.elemSize() * xTileTSize, yTileTSize), false);

            if (!layout.accept(WorkSize(dst.elemSize() * dst.cols, dst.rows), gpu))
                return false;
            
            task.layout = layout.primeLayout();

            if (task.layout.total() < 86)
                return false;

            task.srcTiles = src1.cols / xTileTSize;

            task.async = false;
            task.alpha = alpha;
            task.cn = dst.channels();

            const int buffersNum = 3;
            Offload2dBuffer* buffers[buffersNum] = {&task.src1, &task.src2, &task.dst};
            Mat* mats[buffersNum] = {&src1, &src2, &dst};

            //if (src1.datastart == src2.datastart)
            //    if ((size_t)src1.dataend > (size_t)src2.dataend)
            //        task.src2.autoShare = false;
            //    else
            //        task.src1.autoShare = false;

            for (int i = 0; i < buffersNum; i++) {
                *buffers[i] = Offload2dBuffer(*mats[i]);
                if (!buffers[i]->share())
                    return false;
            }

        } catch (std::exception& e) {
            return false;
        }

        return true;
    }

    using GpuTaskT = MatmultGpuTask;
    const GpuTaskT& advisedGpu() const {
        return task;
    }

private:

    MatmultGpuTask task;
};


class MatmultIgpuWorker {
public:
    MatmultIgpuWorker() {}

    void start(const MatmultGpuTask& _task) throw (std::logic_error) {
        task = &_task;
        
        if (task->cn == 1)
            return start_<16, 16, 1>();

        throw std::logic_error("MatmultIgpuWorker start: unsupported cn");
    }

    void finalize() {
        if (lastTaskId && _GFX_wait(0))
		    throw std::logic_error("MatmultIgpuWorker finalize: _GFX_get_last_error() != GFX_SUCCESS");
        lastTaskId = 0;

        ////c_stdout << "finalize taskId " << lastTaskId << std::endl;
    }
    
private:

    GfxTaskId lastTaskId = 0;

    template<int xTileTSize, int yTileTSize, int cn>
    void start_() throw (std::logic_error) {

        Offload2dBuffer src1(task->src1);
        Offload2dBuffer src2(task->src2);
        Offload2dBuffer dst(task->dst);

        shiftWithOffset(task->layout.offset(), src1, src2, dst);

        auto workRects = task->layout.workRects();

        for (const auto& work : workRects) {
            GfxTaskId taskId = 0 ? 0 : _GFX_offload(&matmult_tiled<float, float, float, float, cn, xTileTSize, yTileTSize >,
                                (const float*)src1.buffer, src1.step,
                                (const float*)src2.buffer, src2.step,
                                (float*)dst.buffer, dst.step,                                                       
                                work.x, work.x + work.width - 1, work.y, work.y + work.height - 1,  
                                task->srcTiles, task->alpha);

            1 ? 0 : matmult_tiled<float, float, float, float, cn, xTileTSize, yTileTSize >(
                                (const float*)src1.buffer, src1.step,
                                (const float*)src2.buffer, src2.step,
                                (float*)dst.buffer, dst.step,                                                       
                                work.x, work.x + work.width - 1, work.y, work.y + work.height - 1,  
                                task->srcTiles, task->alpha);
            if (_GFX_get_last_error() != GFX_SUCCESS) {
                finalize();
                throw std::logic_error("MatmultIgpuWorker start: _GFX_get_last_error() != GFX_SUCCESS");
            }

            lastTaskId = taskId;
        }
    }

    const MatmultGpuTask* task;
};


bool gemm(InputArray _src1, InputArray _src2, double alpha,
                       InputArray _src3, double beta, OutputArray _dst, int flags) {
    Mat src1 = _src1.getMat();
    Mat src2 = _src2.getMat();
    Mat src3 = _src3.getMat();
    
    if (src1.type() != src2.type())
        return false;

    _dst.create(Size(src2.cols, src1.rows), src1.type());
    Mat dst = _dst.getMat();

    {
        GemmAdviser adviser;

        //t2start
        if (src1.datastart == src2.datastart)
            src1 = src1.clone();
        if (src1.datastart == dst.datastart || src2.datastart == dst.datastart)
            return false;
        //t2end("clone       ")
        if (flags) //TODO
            return false;

        if (!adviser.accept(src1, src2, alpha, src3, beta, dst, flags))
            return false;

        const auto& gpuTask = adviser.advisedGpu();

        if (!gpuTask.async) {
            MatmultIgpuWorker gpuWorker;
        

            gpuWorker.start(gpuTask);

        //t2start

            gpuWorker.finalize();

            if (src3.total()) //TODO
                dst += src3 + beta;
        //t2end("wait       ")

            return true;
        }
    }

    return false;
}

}
}
#endif
