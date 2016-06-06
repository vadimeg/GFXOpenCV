#include "../../../core/src/igpu/build_list.hpp"

#ifdef HAVE_IGPU
#include "../../../core/src/igpu/cv_igpu_interface.hpp"
#include "../../../core/src/igpu/igpu_comp_primitives.hpp"
#include "filter2d.hpp"


__declspec(target(gfx_kernel))
void tmpFilter(int* ptr) {
    cilk_for (int yTile = 0; yTile <= 0; yTile++)
        *ptr = 0;
}

namespace cv {

void sepFilter2D_cpu( InputArray _src, OutputArray _dst, int ddepth,
                      InputArray _kernelX, InputArray _kernelY, Point anchor,
                      double delta, int borderType );

void filter2D_cpu( InputArray _src, OutputArray _dst, int ddepth,
                   InputArray _kernel, Point anchor0,
                   double delta, int borderType );

namespace igpu {


#define convolution3_atPixelChannel(x, y) {                                                                  \
        SUM_T sum = (SUM_T) cachedSrc[y - 1][x - cn] * kernel[0] + cachedSrc[y - 1][x] * kernel[1] + cachedSrc[y - 1][x + cn] * kernel[2] +    \
                             cachedSrc[y][x - cn] * kernel[3] +     cachedSrc[y][x] * kernel[4] +     cachedSrc[y][x + cn] * kernel[5] +                          \
                             cachedSrc[y + 1][x - cn] * kernel[6] + cachedSrc[y + 1][x] * kernel[7] + cachedSrc[y + 1][x + cn] * kernel[8] + delta;                       \
        cachedDstRow[x - cn * xRadius] = (T)clamp(sum, minSum, maxSum);                           \
}
#define rowConvolution3_atPixelChannel(x, y) {                                                                  \
        SUM_T sum = (SUM_T) cachedSrc[y][x - cn] * kernel[0] + cachedSrc[y][x] * kernel[1] + cachedSrc[y][x + cn] * kernel[2] + delta;    \
        cachedDstRow[x - cn * xRadius] = (T)clamp(sum, minSum, maxSum);                           \
}
#define colConvolution3_atPixelChannel(x, y) {                                                                  \
        SUM_T sum = (SUM_T) cachedSrc[y - 1][x] * kernel[0] + cachedSrc[y][x] * kernel[1] + cachedSrc[y + 1][x] * kernel[2] + delta;    \
        cachedDstRow[x - cn * xRadius] = (T)clamp(sum, minSum, maxSum);                           \
}

#define convolution_atPixelChannel(x, y) {                                                                  \
        SUM_T sum = delta + __sec_reduce_add(cachedSrc[y - yRadius : yWindowSizePix][x - cn * xRadius : xWindowSizePix : cn] * kernel[:][:]);       \
        cachedDstRow[x - cn * xRadius] = (T)clamp(sum, minSum, maxSum);                           \
}

//#define convolution_atPixelChannel(x, y) {                                                                  \
//        SUM_T sum = delta;                                                                                  \
//        for (int j = 0; j < yWindowSizePix; j++)                                                               \
//            unrolled_for (int i = 0; i < xWindowSizePix; i++)                                                          \
//                sum += cachedSrc[y + j - yRadius][x + cn * (i - xRadius)] * kernel[j * xWindowSizePix + i];       \
//        cachedDstRow[x - cn * xRadius] = (T)clamp(sum, minSum, maxSum);                           \
//}

#define convolution_atPixelChannel_fullUnrolled(x, y) {                                                                  \
        SUM_T sum = delta;                                                                                  \
        for (int j = 0; j < (yWindowSizePix); j++) /*compilation time optimization*/                          \
            for (int i = 0; i < (xWindowSizePix); i++)                                       \
                sum += cachedSrc[y + j - yRadius][x + cn * (i - xRadius)] * kernel[j][i];       \
        cachedDstRow[x - cn * xRadius] = (T)clamp(sum, minSum, maxSum);                           \
}

//#define rowConvolution_atPixelChannel(x, y) {                                                                   \
//        SUM_T sum = (SUM_T) 0;                                                                                     \
//        unrolled_for (int i = 0; i < xWindowSizePix; i++)                                                                \
//            sum += cachedSrc[y][x + cn * (i - xRadius)] * kernel[i];                                           \
//        cachedDstRow[x - cn * xRadius] = (T)clamp(sum + delta, minSum, maxSum);                       \
//}
//
//#define colConvolution_atPixelChannel(x, y) {                                                                \
//        SUM_T sum = (SUM_T) 0;                                                                                    \
//        unrolled_for (int j = 0; j < yWindowSizePix; j++)                                                              \
//            sum += cachedSrc[y + j - yRadius][x] * kernel[j];                                             \
//        cachedDstRow[x - cn * xRadius] = (T)clamp(sum + delta, minSum, maxSum);                           \
//}

enum ConvolutionType { ROW = 1, COL = 2, BOTH = 3 } ;
template< typename T, typename KER_T, typename SUM_T, SUM_T minSum, SUM_T maxSum, int cn, int xWindowSizePix, int yWindowSizePix, int xTileTSize, int yTileTSize >
__declspec(target(gfx_kernel))
void convolution_noBorders_tiled(const uchar* restrict srcptr, int srcStep,
                                uchar* restrict dstptr, int dstStep,
                                //int xAddrLimit, int yAddrLimit,
                                int xItersStart, int xItersEnd, int yItersStart, int yItersEnd,
                                SUM_T delta, const KER_T* restrict kernel_) {
    
    cilk_for (int yTile = yItersStart; yTile <= yItersEnd; yTile++)
        cilk_for (int xTile = xItersStart; xTile <= xItersEnd; xTile++) {
            const int xRadius = xWindowSizePix / 2;
            const int yRadius = yWindowSizePix / 2;
            
            const int xCacheTSize = xTileTSize + 2 * cn * xRadius;
            const int yCacheTSize = yTileTSize + 2 * yRadius;

            const T (* src)[srcStep] = (const T (*)[]) srcptr;
            T (* dst)[dstStep] = (T (*)[]) dstptr;
            
            KER_T kernel[yWindowSizePix][xWindowSizePix];
            
            __assume_aligned(kernel_, 16);
            ((KER_T*)kernel)[0 : yWindowSizePix * xWindowSizePix] = kernel_[0 : yWindowSizePix * xWindowSizePix];
            
            atTile_initCache();
            atTile_loadCache(xTile, yTile, cn * xRadius, yRadius);
            
            for (int y = yRadius; y < yRadius + yTileTSize; y++) {
                //if (yTile * yTileTSize + y - yRadius >= yAddrLimit)
                //    break;

                T cachedDstRow[xTileTSize];
                
                //if (yTile == 10000)
                simd_for (int x = xRadius * cn; x < xRadius * cn + xTileTSize; x++)
                    convolution_atPixelChannel_fullUnrolled(x, y)

                dst[yTile * yTileTSize + y - yRadius][xTile * xTileTSize : xTileTSize] = cachedDstRow[:];

                //memcpy_simple(&dst[yTile * yTileTSize + y][xTile * xTileTSize],
                //        cachedDstRow,
                //        sizeof(cachedDstRow));
            }
        }
}


struct Filter2dGpuTask {
    bool async;

    Offload2dBuffer src;
    Offload2dBuffer intermediate;
    Offload2dBuffer sIntermediate;
    Offload2dBuffer dst;
    //int ddepth;

    bool isSeparated;
    Offload2dBuffer kernel;
    Offload2dBuffer kernelX;
    Offload2dBuffer kernelY;
    
    using LayoutT = OpLayoutsServer::PrimeLayoutT;
    LayoutT layoutSq;

    LayoutT layoutRow;
    LayoutT layoutCol;

    int xWindowSize;
    int yWindowSize;

    //int xAddrLimit;
    //int yAddrLimit;

    const int& windowSize = xWindowSize;

    float delta;

    int cn;
};

#define yTileSizesSq(r)           \
          ((r == 1) ? 12 :     \
           (r == 2) ? 10 :       \
           (r == 3) ? 5 :       \
           (r == 4) ? 3 :       \
           (r == 5) ? 2 : 0)
#define yTileSizesCol(r)           \
          ((r == 1) ? 12 :     \
           (r == 2) ? 10 :       \
           (r == 3) ? 4 :       \
           (r == 4) ? 3 :       \
           (r == 5) ? 2 : 0)
#define yTileSizesRow(r)           \
          ((r == 1) ? 12 :     \
           (r == 2) ? 10 :       \
           (r == 3) ? 4 :       \
           (r == 4) ? 3 :       \
           (r == 5) ? 2 : 0)

class Filter2dBaseAdviser : public Adviser {
public:

    using GpuTaskT = Filter2dGpuTask;
    const GpuTaskT& advisedGpu() const {
        return task;
    }

protected:

    Filter2dBaseAdviser() {}

    bool acceptBase(InputArray _src, OutputArray _dst, int ddepth,
                   bool isSeparated, InputArray _kernel, InputArray _kernelX, InputArray _kernelY, Point anchor,
                   double delta, int borderType) {
        const int d = _kernel.cols();                                                                                                         
        const int r = d / 2;
        const int xD = isSeparated ? _kernelX.cols() : d;                                                                                                         
        const int xR = xD / 2;
        const int yD = isSeparated ? _kernelY.cols() : d;                                                                                                         
        const int yR = yD / 2;

        try {
        
            Mat kernel = _kernel.getMat();
            Mat kernelX = _kernelX.getMat();
            Mat kernelY = _kernelY.getMat();

            if (borderType & BORDER_ISOLATED)
                return false;

            if (isSeparated && anchor.y != yR && anchor.x != xR)
                return false;

            if (!inRange(xD, 3, 11))
                return false;
            if (!inRange(yD, 3, 11))
                return false;

            if (isSeparated) {
                if (kernelX.type() != CV_32FC1 || kernelY.type() != CV_32FC1)  
                    return false;
                if (xD % 2 != 1)
                    return false;
                if (yD % 2 != 1)
                    return false;
                task.xWindowSize = xD;
                task.yWindowSize = yD;
            } else {
                if (kernel.cols != kernel.rows)
                    return false;
                if (kernel.type() != CV_32FC1)   
                    return false;
                if (d % 2 != 1)
                    return false;
                task.xWindowSize = d;
                task.yWindowSize = d;
            }

            if (_src.type() != CV_8UC3)  
                return false;
                                     
            const int cn = _dst.channels();  

            float gpuPart = getGpuPart();

            src = _src.getMat();
            Mat dst = _dst.getMat();

            if (src.datastart == dst.datastart)
                src = src.clone();
        
            //c_stdout << "yD " << yD << endl;
            //c_stdout << "xD " << xD << endl;
            //c_stdout << "anchor " << anchor << endl;
        
            int bottomExt;
            int topExt;

            //int leftExt;
            //int rightExt;

            {
                if (!isSeparated || borderType & BORDER_ISOLATED)
                    bottomExt = topExt = /*rightExt = leftExt = */0;
                else {
                    Size wholeSize;
                    Point ofs;
                    src.locateROI(wholeSize, ofs);

                    Rect roi;
                    roi.y = ofs.y;
                    roi.height = src.rows;
                    //roi.x = ofs.x;
                    //roi.width = src.cols;

                    //c_stdout << "wholeSize.height " << wholeSize.height << " " << roi.y << " " << roi.height << endl;
                    topExt = min(roi.y, anchor.y);
                    bottomExt = min(wholeSize.height - roi.y - roi.height, 2 * yR - anchor.y);
                    //leftExt = min(roi.x, xR);
                    //rightExt = min(wholeSize.width - roi.x - roi.width, xR);
                }
            }

            //c_stdout << "topExt " << topExt << endl;
            //c_stdout << "bottomExt " << bottomExt << endl;

            if (isSeparated) {
                intermediate.create(Size(dst.size().width/* + 2 * xR - leftExt - rightExt*/,
                    dst.size().height + topExt + bottomExt), dst.type());
            }

            //int tmp;
            int rowsNotTaked;
        
            if (isSeparated) {
                const int xTileTSize = 128 / dst.elemSize1();
                const int yTileTSize = yTileSizesRow(xR);

                //c_stdout << "xTileTSize " << xTileTSize << " " << yTileTSize << endl;
            
                WorkerDescriptor gpu(AtomsPerIteration(intermediate.elemSize1() * xTileTSize, yTileTSize), false);
                WorkerDescriptor cpu(AtomsPerIteration(intermediate.elemSize(), 1), true);
        
                if (!layout.accept(WorkSize(intermediate.elemSize() * intermediate.cols, intermediate.rows), gpuPart, gpu, cpu, cpu,
                                   clarifyBordersSize(BordersSize(dst.elemSize() * anchor.x, 0,
                                                                  dst.elemSize() * (2 * xR - anchor.x), 0),
                                                      src, borderType)))
                    return false;

                rowsNotTaked = max(intermediate.rows - layout.primeLayout().workRects()[0].height * yTileTSize, 0);
                //c_stdout << "rowsNotTaked " << rowsNotTaked << endl;

                //tmp = layout.borderLayout().workRects()[6].height;
                //tmp.y -= topExt;
                //tmp.y = max(0, tmp.y);
                task.layoutRow = layout.primeLayout();

                //if (task.layoutRow.total() < 86)
                //    return false;
            }
        
            Mat sIntermediate;
            if (isSeparated) {
                sIntermediate = intermediate(Rect(0, topExt, dst.cols, dst.rows));
            }
        
            if (isSeparated) {
                const int xTileTSize = 128;
                const int yTileTSize = yTileSizesCol(yR);

                //c_stdout << "xTileTSize " << xTileTSize << " " << yTileTSize << endl;
            
                WorkerDescriptor gpu(AtomsPerIteration(dst.elemSize1() * xTileTSize, yTileTSize), false);
                WorkerDescriptor cpu(AtomsPerIteration(dst.elemSize(), 1), true);

                BordersSize b = clarifyBordersSize(BordersSize(dst.elemSize() * anchor.x, anchor.y,
                                                             dst.elemSize() * (2 * xR - anchor.x), 2 * yR - anchor.y + rowsNotTaked), sIntermediate, borderType);
                BordersSize bSrc = clarifyBordersSize(BordersSize(dst.elemSize() * anchor.x, anchor.y,
                                                             dst.elemSize() * (2 * xR - anchor.x), 2 * yR - anchor.y + rowsNotTaked), src, borderType);
                b.left = bSrc.left;
                b.right = bSrc.right;
        
                if (!layout.accept(WorkSize(dst.elemSize() * dst.cols, dst.rows), gpuPart, gpu, cpu, cpu, b))
                    return false;

                task.layoutCol = layout.primeLayout();

                //if (task.layoutCol.total() < 86)
                //    return false;
            }

            {
                const int xTileTSize = 128;
                const int yTileTSize = !isSeparated ? yTileSizesSq(r) : yTileSizesCol(yR);

                //c_stdout << "xTileTSize " << xTileTSize << " " << yTileTSize << endl;
            
                WorkerDescriptor gpu(AtomsPerIteration(dst.elemSize1() * xTileTSize, yTileTSize), false);
                WorkerDescriptor cpu(AtomsPerIteration(dst.elemSize(), 1), true);

                BordersSize b = clarifyBordersSize(BordersSize(dst.elemSize() * anchor.x, anchor.y,
                                                             dst.elemSize() * (2 * xR - anchor.x), (isSeparated ?
                                                                 (dst.rows - task.layoutCol.workRects()[0].height * yTileSizesCol(yR)) : 0) +
                                                                 2 * yR - anchor.y), src, borderType);
            
                //c_stdout << "b l " << b.left << endl;
                //c_stdout << "b r " << b.right << endl;
                //c_stdout << "b t " << b.top << endl;
                //c_stdout << "b b " << b.bottom << endl;
        
                if (!layout.accept(WorkSize(dst.elemSize() * dst.cols, dst.rows), gpuPart, gpu, cpu, cpu, b))
                    return false;

                if (!isSeparated) {
                    task.layoutSq = layout.primeLayout();
                    //if (task.layoutSq.total() < 86)
                    //    return false;
                }
            }

            task.async = false;
            task.delta = (float) delta;
            task.cn = cn;
        
            const int buffersNumAny = 5;
            Offload2dBuffer* buffers[buffersNumAny] = {&task.src, &task.dst, &task.intermediate, &task.kernelX, &task.kernelY};
            Mat* mats[buffersNumAny] = {&src, &dst, &intermediate, &kernelX, &kernelY};
        
            int buffersNum = isSeparated ? buffersNumAny : 3;
            if (!isSeparated) {
                buffers[buffersNum - 1] = &task.kernel;
                mats[buffersNum - 1] = &kernel;
            }
        
            for (int i = 0; i < buffersNum; i++)
                *buffers[i] = Offload2dBuffer(*mats[i]);
        
            if (kernelX.datastart == kernelY.datastart)
                if (xD > yD)
                    task.kernelY.autoShare = false;
                else
                    task.kernelX.autoShare = false;

            for (int i = 0; i < buffersNum; i++)
                if (!buffers[i]->share())
                    return false;

            task.src.buffer = (uchar*)task.src.buffer - topExt * task.src.stepBytes;
            task.src.buffer = (uchar*)task.src.buffer + dst.elemSize() * (xR - anchor.x) + task.src.stepBytes * (yR - anchor.y);

            task.sIntermediate = task.intermediate;
            task.sIntermediate.autoShare = false;
            task.sIntermediate.buffer = sIntermediate.data;

            task.isSeparated = isSeparated;

        } catch (std::exception& e) {
            return false;
        }

        return true;
    }

private:
    Mat src;
    Mat intermediate;

    Filter2dGpuTask task;
};

class Filter2dAdviser : public Filter2dBaseAdviser {
public:
    Filter2dAdviser() {}

    bool accept(InputArray _src, OutputArray _dst, int ddepth,
                   InputArray _kernel, Point anchor0,
                   double delta, int borderType) {
        return acceptBase(_src, _dst, ddepth, false,
                                             _kernel, Mat(), Mat(), anchor0, delta, borderType);
    }
};

class SepFilter2dAdviser : public Filter2dBaseAdviser {
public:
    SepFilter2dAdviser() {}

    bool accept(InputArray _src, OutputArray _dst, int ddepth,
                   InputArray _kernelX, InputArray _kernelY, Point anchor0,
                   double delta, int borderType) {
        return acceptBase(_src, _dst, ddepth, true,
                                             Mat(), _kernelX, _kernelY, anchor0, delta, borderType);
    }
};

class Filter2dIgpuWorker {
public:
    Filter2dIgpuWorker() {}

    void start(const Filter2dGpuTask& _task) throw (std::logic_error) {
        task = &_task;
        
        if (task->isSeparated) {
#ifdef SEPFILTER2D_BUILD
            if (task->cn == 3) {
                const int cn = 3;
           
                return startSep<128, cn>();
            }
#endif
        } else {
#ifdef FILTER2D_BUILD
            if (task->cn == 3) {
                const int cn = 3;
                return startWithType<128, cn, BOTH>();
            }
#endif
        }
        
        throw std::logic_error("Filter2dIgpuWorker start: unsupported params");
    }
    
    void finalize() {
        if (lastTaskId && _GFX_wait(lastTaskId))
		    throw std::logic_error("Filter2dIgpuWorker finalize: _GFX_get_last_error() != GFX_SUCCESS");
        
        lastTaskId = 0;
        //c_stdout << "finalize taskId " << lastTaskId << std::endl;
    }
    
private:
    GfxTaskId lastTaskId = 0;
    
    enum ConvolutionType { ROW = 1, COL = 2, BOTH = 3 } ;

    template<int xTileTSize, int cn>
    void startSep() {
        startWithType<xTileTSize, cn, ROW>();
        //finalize();
        startWithType<xTileTSize, cn, COL>();
    }

    template<int xTileTSize, int cn, ConvolutionType type>
    void startWithType() throw (std::logic_error) {
        if (task->cn == 3) {
            const int cn = 3;

            const int d = type == BOTH ? task->windowSize :
                          type == ROW ? task->xWindowSize :
                          task->yWindowSize;
            switch(d) {
                case 3: return start_<3, 128, cn, type>();
                case 5: return start_<5, 128, cn, type>();
                case 7: return start_<7, 128, cn, type>();
                case 9: return start_<9, 128, cn, type>();
                case 11: return start_<11, 128, cn, type>();
            }
        }
        
        throw std::logic_error("Filter2dIgpuWorker start: unsupported params");
    }


    template<int d, int xTileTSize, int cn, ConvolutionType type>
    void start_() throw (std::logic_error) {
        using SRC_T = uchar;
        using DST_T = uchar;
        using KER_T = float;
        using SUM_T = KER_T;

        //const int xTileTSize = xTileTSize / sizeof(SRC_T);
        
        const int r = d / 2;
        const int yTileTSize = type == BOTH ? yTileSizesSq(r) :
                               type == ROW ? yTileSizesRow(r) :
                               yTileSizesCol(r);
            //c_stdout << "xTileTSize " << xTileTSize << " " << yTileTSize << endl;

        auto& layout = type == BOTH ? task->layoutSq :
                       type == ROW ? task->layoutRow :
                       task->layoutCol;
        
        Offload2dBuffer src(type == COL ? task->sIntermediate : task->src);
        Offload2dBuffer dst(type == ROW ? task->intermediate : task->dst);

            //c_stdout << "type " << type << endl;
            //c_stdout << "layout.offset() " << layout.offset() << endl;
        shiftWithOffset(layout.offset(), src, dst);

        const SUM_T minSum = (SUM_T) 0;
        const SUM_T maxSum = (SUM_T) 255;

        const float* kernel = (const float*) (type == BOTH ? task->kernel.buffer :
                                              type == ROW ? task->kernelX.buffer :
                                              task->kernelY.buffer);

        auto workRects = layout.workRects();
        for (const auto& work : workRects) {
            //c_stdout << "work " << work << endl;
            if (!(work.width * work.height))
                continue;
            GfxTaskId taskId = 0 ? 0 : _GFX_offload(&convolution_noBorders_tiled<SRC_T, KER_T, SUM_T, 0.f, 255.f, cn, type & ROW ? d : 1, type & COL ? d : 1, xTileTSize, yTileTSize>,
                                    (const SRC_T*)src.buffer, src.step,
                                    (DST_T*)dst.buffer, dst.step,
                                    work.x, work.x + work.width - 1, work.y, work.y + work.height - 1,
                                    (type == ROW ? 0.f : task->delta) + 0.5f, kernel);

            if (0) convolution_noBorders_tiled<SRC_T, KER_T, SUM_T, 0.f, 255.f, cn, type & ROW ? d : 1, type & COL ? d : 1, xTileTSize, yTileTSize>(
                                    (const SRC_T*)src.buffer, src.step,
                                    (DST_T*)dst.buffer, dst.step,
                                    work.x, work.x + work.width - 1, work.y, work.y + work.height - 1,
                                    (type == ROW ? 0.f : task->delta) + 0.5f, kernel);
            if (_GFX_get_last_error() != GFX_SUCCESS) {
                finalize();
                throw std::logic_error("Filter2dIgpuWorker start: _GFX_get_last_error() != GFX_SUCCESS");
            }

            lastTaskId = taskId;
        }
    }

    const Filter2dGpuTask* task;
};



bool filter2D( InputArray _src, OutputArray _dst, int ddepth,
               InputArray _kernel, Point anchor0,
               double delta, int borderType ) {
    Mat src = _src.getMat();

    Mat kernel = _kernel.getMat();
    
    if( ddepth < 0 )
        ddepth = src.depth();
    
    _dst.create(src.size(), CV_MAKETYPE(ddepth, src.channels()));
    Mat dst = _dst.getMat();
 
    Point anchor = normalizeAnchor(anchor0, kernel.size());
 
    return performHeteroTaskWithCvFunc(Filter2dAdviser(), Filter2dIgpuWorker(), &cv::filter2D_cpu, src, dst, ddepth, _kernel, anchor, delta, borderType);
}


bool sepFilter2D( InputArray _src, OutputArray _dst, int ddepth,
               InputArray _kernelX, InputArray _kernelY, Point anchor0,
               double delta, int borderType ) {
    Mat src = _src.getMat();
    
    Mat kernelX = _kernelX.getMat();
    Mat kernelY = _kernelY.getMat();
    
    if( ddepth < 0 )
        ddepth = src.depth();

    if (ddepth != src.depth())
        return false;

    _dst.create(src.size(), CV_MAKETYPE(ddepth, src.channels()));
    Mat dst = _dst.getMat();

    if (!makeMatVec(kernelX) || !makeMatVec(kernelY))
        return false; 

    //if (kernelX.cols == kernelY.cols)
    //    return filter2D(_src, _dst, ddepth, gemm(kernelX, transpose(kernelY), anchor0, delta, borderType);
 
    Point anchor = normalizeAnchor(anchor0, Size(kernelX.cols, kernelY.cols));

    //{
    //    const int xD = _kernelX.cols();                                                                                                         
    //    const int xR = xD / 2;
    //    const int yD = _kernelY.rows();                                                                                                         
    //    const int yR = yD / 2;
    //
    //    Point p1(0, yR - anchor.y);
    //    Point p2(src.cols, src.rows - yR + anchor.y);
    //    if (!igpu::filter2D(src(Rect(p1, p2)), dst(Rect(p1, p2)), ddepth, _kernelX, Point(anchor.x, 0), 0.0, borderType))
    //        return false;
    //    Point p3(xR - anchor.x, 0);
    //    Point p4(src.cols - xR + anchor.x, src.rows);
    //    if (!igpu::filter2D(src, dst, ddepth, _kernelY, Point(0, anchor.y), delta, borderType))
    //        cv::filter2D(src, dst, ddepth, _kernelY, Point(0, anchor.y), delta, borderType);
    //}
 
    return performHeteroTaskWithCvFunc(SepFilter2dAdviser(), Filter2dIgpuWorker(), &cv::sepFilter2D_cpu, src, dst, ddepth, kernelX, kernelY, anchor, delta, borderType);
}


}
}
#endif
