#include "../../../core/src/igpu/build_list.hpp"
//#include <mathimf.h>
#ifdef BILATERAL_BUILD
#include "../../../core/src/igpu/cv_igpu_interface.hpp"
#include "../../../core/src/igpu/igpu_comp_primitives.hpp"


#include <memory>

__declspec(target(gfx_kernel))
void tmpBil(int* ptr) {
    cilk_for (int yTile = 0; yTile <= 0; yTile++)
        *ptr = 0;
}

namespace cv {

void bilateralFilter_cpu( InputArray _src, OutputArray _dst, int d,
                      double sigmaColor, double sigmaSpace,
                      int borderType );

namespace igpu {

#define bilateralFilter_atPixel(x, y) {                                                            \
        SUM_T sum[cn];                                                                             \
        SUM_T wsum = (SUM_T)0;                                                                     \
        sum[:] = (SUM_T)0;                                                                     \
        SUM_T val0[cn];                                                                                             \
        for (int c = 0; c < cn; c++) val0[c] = cachedSrc[y][(cn * x) + c];                                                  \
        for (int k = 0; k < maxK; k++) {                                                            \
            SUM_T val[cn];                                                                           \
            SUM_T diff = (SUM_T) 0;                                                                    \
            for (int c = 0; c < cn; c++) {                                               \
                val[c] = (SUM_T) cachedSrc[y + (int)space_ofs[2 * k + 1]][cn * (x + (int)space_ofs[2 * k]) + c];                                     \
                diff += static_abs(val[c] - val0[c]);                                    \
            }                                                                                  \
            const float w = space_weight[k] * expf(diff * diff * gauss_color_coeff);             \
            wsum += w;                                                                             \
            for (int c = 0; c < cn; c++) sum[c] += val[c] * w;                                                              \
        }                                                                                          \
        for (int c = 0; c < cn; c++) cachedDstRow[(cn * (x - xRadius)) + c] = (T)(sum[c] / wsum + (SUM_T) 0.5);             \
}

//#define bilateralFilter_atPixel(x, y) {                                           \
//        SUM_T sum[cn];                                                                             \
//        SUM_T wsum = (SUM_T)0;                                                                     \
//        sum[:] = (SUM_T)0;                                                                     \
//        SUM_T val0[cn];                                                                                \
//        for (int c = 0; c < cn; c++)                                                               \
//            val0[c] = cachedSrc[y][(cn * x) + c];                                                       \
//        for (int k = 0; k < maxK; k++) {                                                           \
//            float diff = 0.f;                                                                      \
//            SUM_T val[cn];                                                                         \
//            for (int c = 0; c < cn; c++) {                                                         \
//                loadSrc((x + (int)space_ofs[2 * k + 1]) * cn + c, y + (int)space_ofs[2 * k], val[c]);         \
//                diff += (float)abs(val[c] - val0[c]);                                              \
//            }                                                                                      \
//            float w = space_weight[k] * expf(diff * diff * gauss_color_coeff);             \
//            wsum += w;                                                                             \
//            sum[:] += val[:] * w;                                                               \
//        }                                                                                          \
//        for (int c = 0; c < cn; c++) {                                                               \
//            const T chan = (T)(sum[c] / wsum);                                                    \
//            cachedDstRow[(cn * (x - xRadius)) + c] = chan;                                                      \
//        }                                                                                           \
//}

const int yyyy = 4;

#define circleSize(r)           \
          (((r) == 1) ? 5 :     \
           ((r) == 2) ? 13 :       \
           ((r) == 3) ? 29 :       \
           ((r) == 4) ? 49 :       \
           ((r) == 5) ? 81 :       \
           ((r) == 6) ? 113 :       \
           ((r) == 7) ? 149 :       \
           ((r) == 8) ? 197 :       \
           ((r) == 9) ? 253 :       \
           ((r) == 10) ? 317 :       \
           ((r) == 11) ? 377 :       \
           ((r) == 12) ? 441 : -1)

template< typename T, typename SUM_T, int cn, int d, int xTilePixSize, int yTilePixSize >
void bilateralFilter_noBorders_tiled_cpu(const uchar* restrict srcptr, int srcStep,
                                uchar* restrict dstptr, int dstStep,
                                int leftItersBorder, int rightItersBorder, int topItersBorder, int bottomItersBorder,
                                const float* restrict color_weight, const float* restrict space_weight, const char* restrict space_ofs) {

    cilk_for (int yTile_ = topItersBorder; yTile_ <= bottomItersBorder; yTile_++)
        cilk_for (int xTile = leftItersBorder; xTile <= rightItersBorder; xTile++) {
            const int xWindowPixSize = d;
            const int yWindowPixSize = d;
            const int xRadius = xWindowPixSize / 2;
            const int yRadius = yWindowPixSize / 2;
            
            const int maxK = circleSize(d / 2);

            const int xCacheTSize = (xTilePixSize + 2 * xRadius) * cn;
            const int yCacheTSize = yTilePixSize + 2 * yRadius;

            const T (* src)[srcStep] = (const T (*)[]) srcptr;
            T (* dst)[dstStep] = (T (*)[]) dstptr;

            for (int yIter = 0; yIter < yyyy; yIter++) {
                const int yTile = yTile_ * yyyy + yIter;
                
                for (int y = yTile * yTilePixSize; y < (yTile + 1) * yTilePixSize; y++) {
                    T cachedDstRow[cn * xTilePixSize];
                    simd_for (int x = xTile * xTilePixSize; x < (xTile + 1) * xTilePixSize; x++) {
                        SUM_T sum[cn];
                        SUM_T wsum = (SUM_T)0;
                        for (int c = 0; c < cn; c++)
                            sum[c] = (SUM_T)0;
                        int val0[cn];
                        for (int c = 0; c < cn; c++)
                            val0[c] = src[y][cn * x + c];
                        #pragma unroll(5)
                        for (int k = 0; k < maxK; k++) {
                            T val[cn];
                            int diff = 0;
                            for (int c = 0; c < cn; c++) {
                                val[c] = (T) src[y + space_ofs[2 * k + 1]][cn * (x + space_ofs[2 * k]) + c];
                            }
                            for (int c = 0; c < cn; c++) {
                                diff += static_abs((int)val[c] - (int)val0[c]);
                            }
                            float w = space_weight[k] * color_weight[diff];
                            wsum += w;
                            for (int c = 0; c < cn; c++)
                                sum[c] += val[c] * w;
                        }
                        for (int c = 0; c < cn; c++)
                            cachedDstRow[cn * (x - xTile * xTilePixSize) + c] = (T)(sum[c] / wsum + (SUM_T) 0.5);
                    }
                    dst[y][xTile * cn * xTilePixSize : cn * xTilePixSize] = cachedDstRow[:]; 
                }
            }
        }
}

template< typename T, typename SUM_T, int cn, int d, int xTilePixSize, int yTilePixSize >
__declspec(target(gfx_kernel))
void bilateralFilter_noBorders_tiled(const uchar* restrict srcptr, int srcStep,
                                uchar* restrict dstptr, int dstStep,
                                int leftItersBorder, int rightItersBorder, int topItersBorder, int bottomItersBorder,
                                int maxK_v, float gauss_color_coeff, const float* restrict _space_weight, const char* restrict _space_ofs) {

    cilk_for (int yTile_ = topItersBorder; yTile_ <= bottomItersBorder; yTile_++)
        cilk_for (int xTile = leftItersBorder; xTile <= rightItersBorder; xTile++) {
            const int xWindowPixSize = d;
            const int yWindowPixSize = d;
            const int xRadius = xWindowPixSize / 2;
            const int yRadius = yWindowPixSize / 2;
            
            const int maxK_c = circleSize(d / 2);
            const int maxK = d == 3 ? maxK_v : maxK_c;

            const int xTileTSize = xTilePixSize * cn;
            const int yTileTSize = yTilePixSize;
            const int xCacheTSize = (xTilePixSize + 2 * xRadius) * cn;
            const int yCacheTSize = yTilePixSize + 2 * yRadius;
            
            char space_ofs[2 * maxK_c];
            float space_weight[maxK_c];
            space_ofs[0 : 2 * maxK_c] = _space_ofs[0 : 2 * maxK_c];
            space_weight[0 : maxK_c] = _space_weight[0 : maxK_c];

            const T (* src)[srcStep] = (const T (*)[]) srcptr;
            T (* dst)[dstStep] = (T (*)[]) dstptr;

            const int yTile = yTile_ * yyyy;

            atTile_initCache();
            atTile_loadCache(xTile, yTile, cn * xRadius, yRadius);

            for (int y = yRadius; y < yRadius + yTilePixSize; y++) {
                T cachedDstRow[cn * xTilePixSize];
                simd_for (int x = xRadius; x < xRadius + xTilePixSize; x++)
                    bilateralFilter_atPixel(x, y);

                //memcpy_(&dst[yTile * yTilePixSize + y - yRadius][xTile * cn * xTilePixSize], cachedDstRow, sizeof(cachedDstRow));

                dst[yTile * yTilePixSize + y - yRadius][(xTile * cn * xTilePixSize) : cn * xTilePixSize] = cachedDstRow[:];            
            }
            for (int yIter = 1; yIter < yyyy; yIter++) {
                cachedSrc[0 : yRadius * 2][:] = cachedSrc[(yCacheTSize - yRadius * 2) : yRadius * 2][:];
                const int yTile = yTile_ * yyyy + yIter;
                atTilePart_loadCache(xTile, yTile, cn * xRadius, yRadius, yRadius * 2, yCacheTSize - 1);
            
                for (int y = yRadius; y < yRadius + yTilePixSize; y++) {
                    T cachedDstRow[cn * xTilePixSize];
                    simd_for (int x = xRadius; x < xRadius + xTilePixSize; x++)
                        bilateralFilter_atPixel(x, y);

                    //memcpy_(&dst[yTile * yTilePixSize + y - yRadius][xTile * cn * xTilePixSize], cachedDstRow, sizeof(cachedDstRow));

                    dst[yTile * yTilePixSize + y - yRadius][(xTile * cn * xTilePixSize) : cn * xTilePixSize] = cachedDstRow[:];            
                }
            }
        }
}


struct BilateralGpuTask {
    bool async;

    Offload2dBuffer src;
    Offload2dBuffer dst;

    Offload2dBuffer ospace;
    Offload2dBuffer wspace;
    Offload2dBuffer cspace;
    
    using LayoutT = OpLayoutsServer::PrimeLayoutT;
    LayoutT layoutGpu;
    LayoutT layoutCpu;
    
    int cn;

    int maxK;

    int d;
    float gauss_color_coeff;
    float gauss_space_coeff;
};

#define yTileSizes(r)           \
          (((r) == 1) ? 10 :     \
           ((r) == 2) ? 7 :       \
           ((r) == 3) ? 6 :       \
           ((r) == 4) ? 5 :       \
           ((r) == 5) ? 2 :       \
           ((r) == 6) ? 1 :       \
           ((r) == 7) ? 1 :       \
           ((r) == 8) ? 1 :       \
           ((r) == 9) ? 1 :       \
           ((r) == 10) ? 1 :       \
           ((r) == 11) ? 1 :       \
           ((r) == 12) ? 1 :       \
           ((r) == 13) ? 1 : -1)

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
        if (!inRange(d, 3, 15))
            return false;
    
        if (_src.type() != CV_8UC1 && _src.type() != CV_8UC3)  
            return false;

        if (borderType & BORDER_ISOLATED)
            return false;
        
        const int cn = _dst.channels();  

        float gpuPart = getGpuPart();
        
        try {

            float gauss_color_coeff = -0.5f / (sigmaColor * sigmaColor);
            float gauss_space_coeff = -0.5f / (sigmaSpace * sigmaSpace);
            int maxK = circleSize(r);

            space_o_m.create(Size(maxK, 1), CV_8UC2);
            space_w_m.create(Size(maxK, 1), CV_32FC1);
            space_c_m.create(Size(cn * 256, 1), CV_32FC1);
            //c_stdout << "maxK " << maxK << endl;
            task.maxK = maxK;

            for( int k = 0, int i = -r; i <= r; i++ )
                for( int j = -r; j <= r; j++ ) {
                    float rr = std::sqrt((float)i * i + (float)j * j);
                    if ( rr > r )
                        continue;
                    ((float*)space_w_m.data)[k] = (float)std::expf(rr * rr * gauss_space_coeff);
                    space_o_m.data[2 * k] = i;
                    space_o_m.data[2 * k++ + 1] = j;
                }

            simd_for (int i = 0; i < cn * 256; i++)
                ((float*)space_c_m.data)[i] = (float)std::expf(i * i * gauss_color_coeff);

            Mat src = _src.getMat();
            Mat dst = _dst.getMat();
            task.async = false;
            task.d = d;
            task.cn = cn;

            //c_stdout << "cn " << cn << endl;
            //c_stdout << "d " << d << endl;

            task.gauss_color_coeff = -0.5f / (sigmaColor * sigmaColor);
            task.gauss_space_coeff = -0.5f / (sigmaSpace * sigmaSpace);
        
            {
                const int xTilePixSize = 32;
                const int yTilePixSize = yTileSizes(r) * yyyy;

                //c_stdout << "xTilePixSize " << xTilePixSize << " " << yTilePixSize << endl;
            
                WorkerDescriptor gpu(AtomsPerIteration(dst.elemSize() * xTilePixSize, yTilePixSize), false);
                WorkerDescriptor cpu(AtomsPerIteration(dst.elemSize() * 64, yTilePixSize), false);
                WorkerDescriptor border(AtomsPerIteration(dst.elemSize(), 1), true);
        
                if (!layout.accept(WorkSize(dst.elemSize() * dst.cols, dst.rows), gpuPart, gpu, cpu, border,
                                    clarifyBordersSize(BordersSize(dst.elemSize() * r, r), src, borderType)))
                    return false;

                task.layoutGpu = layout.primeLayout();
                task.layoutCpu = layout.secondLayout();
            }

            //if (task.layoutGpu.total() < 86)
            //    return false;

            const int buffersNum = 5;
            Offload2dBuffer* buffers[buffersNum] = {&task.src, &task.dst, &task.ospace, &task.wspace, &task.cspace};
            Mat* mats[buffersNum] = {&src, &dst, &space_o_m, &space_w_m, &space_c_m};

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

    using GpuTaskT = BilateralGpuTask;
    const GpuTaskT& advisedGpu() const {
        return task;
    }
    

private:
    Mat space_o_m;
    Mat space_w_m;
    Mat space_c_m;

    BilateralGpuTask task;
};


class BilateralFilterIgpuWorker {
public:
    BilateralFilterIgpuWorker() {}
    
    void start(const BilateralGpuTask& _task) throw (std::runtime_error) {
        task = &_task;

        if (task->cn == 3) 
            return startWithCn<3>();
        if (task->cn == 1) 
            return startWithCn<1>();
        
        throw std::runtime_error("BilateralFilterIgpuWorker start: unsupported cn");

    }
    
    void finalize() {
        if (lastTaskId && _GFX_wait(lastTaskId))
		    throw std::runtime_error("BilateralFilterIgpuWorker finalize: _GFX_get_last_error() != GFX_SUCCESS");
        
        lastTaskId = 0;
        //c_stdout << "finalize taskId " << lastTaskId << std::endl;
    }
    
private:
    GfxTaskId lastTaskId = 0;

    template<int cn>
    void startWithCn() throw (std::runtime_error) {
        switch(task->d) {
            case 3: startGpu<3, 32, cn>();
                    return startCpu<3, 64, cn>();
            case 5: startGpu<5, 32, cn>();
                    return startCpu<5, 64, cn>();
            case 7: startGpu<7, 32, cn>();
                    return startCpu<7, 64, cn>();
            case 9: startGpu<9, 32, cn>();
                    return startCpu<9, 64, cn>();
            case 11: startGpu<11, 32, cn>();
                    return startCpu<11, 64, cn>();
            case 13: startGpu<13, 32, cn>();
                    return startCpu<13, 64, cn>();
            case 15: startGpu<15, 32, cn>();
                    return startCpu<15, 64, cn>();
        }
        throw std::runtime_error("BilateralFilterIgpuWorker start: unsupported params");
    }

    template<int d, int xTilePixSize, int cn>
    void startGpu() throw (std::runtime_error) {
        using SRC_T = uchar;
        using DST_T = uchar;
		
        const int r = d / 2;
        const int yTilePixSize = yTileSizes(r);
            //c_stdout << "xTilePixSize " << xTilePixSize << " " << yTilePixSize << endl;
        
        Offload2dBuffer src(task->src);
        Offload2dBuffer dst(task->dst);
    
            //c_stdout << "task->layoutGpu.offset() " << task->layoutGpu.offset() << endl;
    
        shiftWithOffset(task->layoutGpu.offset(), src, dst);
    
        auto workRects = task->layoutGpu.workRects();
    
        for (const auto& work : workRects) {
            //c_stdout << "work " << work << endl;
            if (!(work.width * work.height))
                continue;
            GfxTaskId taskId = _GFX_offload(&bilateralFilter_noBorders_tiled<SRC_T, float, cn, d, xTilePixSize, yTilePixSize>,
                                 (const SRC_T*)src.buffer, src.step,
                                 (DST_T*)dst.buffer, dst.step,
                                 work.x, work.x + work.width - 1, work.y, work.y + work.height - 1,
                                 task->maxK, task->gauss_color_coeff, (float*)task->wspace.buffer, (char*)task->ospace.buffer);

            if (_GFX_get_last_error() != GFX_SUCCESS) {
                finalize();
                throw std::runtime_error("BilateralFilterIgpuWorker start: _GFX_get_last_error() != GFX_SUCCESS");
            }
    
            lastTaskId = taskId;
        }
    }

    template<int d, int xTilePixSize, int cn>
    void startCpu() throw (std::runtime_error) {
        using SRC_T = uchar;
        using DST_T = uchar;
		
        const int r = d / 2;
        const int yTilePixSize = yTileSizes(r);
            //c_stdout << "xTilePixSize " << xTilePixSize << " " << yTilePixSize << endl;
        
        Offload2dBuffer src(task->src);
        Offload2dBuffer dst(task->dst);

            //c_stdout << "task->layoutCpu.offset() " << task->layoutCpu.offset() << endl;

        shiftWithOffset(task->layoutCpu.offset(), src, dst);

        auto workRects = task->layoutCpu.workRects();

        for (const auto& work : workRects) {
            //c_stdout << "work " << work << endl;
            if (!(work.width * work.height))
                continue;

            bilateralFilter_noBorders_tiled_cpu<SRC_T, float, cn, d, xTilePixSize, yTilePixSize>(
                                 (const SRC_T*)src.buffer, src.step,
                                 (DST_T*)dst.buffer, dst.step,
                                 work.x, work.x + work.width - 1, work.y, work.y + work.height - 1,
                                 (float*)task->cspace.buffer, (float*)task->wspace.buffer, (char*)task->ospace.buffer);
        }
    }

    const BilateralGpuTask* task;
};


bool bilateralFilter( InputArray _src, OutputArray _dst, int d,
                      double sigmaColor, double sigmaSpace,
                      int borderType ) {
    Mat src = _src.getMat();

    if( sigmaColor <= 0 )
        sigmaColor = 1;
    if( sigmaSpace <= 0 )
        sigmaSpace = 1;

    if( d <= 0 )
        d = max(cvRound(sigmaSpace * 1.5), 1) * 2 + 1;

    _dst.create(src.size(), src.type());
    
    Mat dst = _dst.getMat();

    if (src.datastart == dst.datastart)
        return false;
        
    return performHeteroTaskWithCvFunc_onlyBorders(BilateralAdviser(), BilateralFilterIgpuWorker(), &cv::bilateralFilter_cpu, src, dst,
                             d, sigmaColor, sigmaSpace, borderType);
}

}
}

#endif
