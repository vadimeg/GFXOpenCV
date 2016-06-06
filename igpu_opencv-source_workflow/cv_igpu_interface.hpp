#pragma once

#include <opencv2/opencv.hpp>

#include <gfx/gfx_rt.h>
#include <cilk/cilk.h>
#include <algorithm>
#include <array>

#include "debug.hpp"

//using cv::Rect;
//using cv::Point;

namespace cv {
namespace igpu {

#define CV_IGPU_RUN(condition, func)              \
    if ((condition) && func)                      \
        return;

template <typename T>
T div(T a, T b, bool isTop) {
    return (a / b) + (isTop ? !!(a % b) : 0);
}

using WorkSize = Size;
using AtomBytesSize = Size;
using AtomsPerIteration = Size;
using ItersStart = Point;

struct WorkerDescriptor {
    WorkerDescriptor(AtomsPerIteration _bytesPerIteration = Size(1, 1), bool _considersBorders = false)
     : bytesPerIteration(_bytesPerIteration), considersBorders(_considersBorders)
    {}

    AtomsPerIteration bytesPerIteration;
    bool considersBorders;
};

class SolidOperationLayout {
public:
    SolidOperationLayout() {}

    SolidOperationLayout(WorkerDescriptor worker, Rect _workRect) {
        assert(worker.considersBorders ||
               ((!_workRect.width % worker.bytesPerIteration.width) &&
               (!_workRect.height % worker.bytesPerIteration.height)));
        this->_workRects[0].x = div(_workRect.x, worker.bytesPerIteration.width, false);
        this->_workRects[0].y = div(_workRect.y, worker.bytesPerIteration.height, false);
        this->_workRects[0].width = div(_workRect.width, worker.bytesPerIteration.width, true);
        this->_workRects[0].height = div(_workRect.height, worker.bytesPerIteration.height, true);

        offset_.width = _workRect.x;
        offset_.height = _workRect.y;
    }

    using WorkRectsT = std::array<Rect, 1>;
    const WorkRectsT& workRects() const {
        return _workRects;
    }
    const Size offset() const {
        return offset_;
    }
protected:
    Size offset_;
    std::array<Rect, 1> _workRects;
};

class BorderOperationLayout {
public:
    BorderOperationLayout() {}

    BorderOperationLayout(WorkerDescriptor border, Size wholeSize, Rect primeSpace, Rect secondSpace) {
        _workRects[0] = Rect(Point(0, 0), Point(wholeSize.width, primeSpace.y));
        _workRects[1] = Rect(Point(0, primeSpace.y), Point(primeSpace.x, primeSpace.y + primeSpace.height));
        _workRects[2] = Rect(Point(primeSpace.width + primeSpace.x, primeSpace.y), Point(wholeSize.width, primeSpace.y + primeSpace.height));
        
        if (!(secondSpace.width * secondSpace.height))
            secondSpace.y = primeSpace.y + primeSpace.height;
        _workRects[3] = Rect(Point(0, primeSpace.y + primeSpace.height), Point(wholeSize.width, secondSpace.y));

        _workRects[4] = Rect(Point(0, secondSpace.y), Point(secondSpace.x, secondSpace.y + secondSpace.height));
        _workRects[5] = Rect(Point(secondSpace.width + secondSpace.x, secondSpace.y), Point(wholeSize.width, secondSpace.y + secondSpace.height));
        _workRects[6] = Rect(Point(0, secondSpace.y + secondSpace.height), Point(wholeSize.width, wholeSize.height));

        for (auto& w : _workRects) {
            if (border.bytesPerIteration.width) {
                w.x /= border.bytesPerIteration.width;
                w.width = div(w.width, border.bytesPerIteration.width, true);
            }
            if (border.bytesPerIteration.height) {
                w.y /= border.bytesPerIteration.height;
                w.height = div(w.height, border.bytesPerIteration.height, true);
            }
        }
    }

    using WorkRectsT = std::array<Rect, 7>;
    const WorkRectsT& workRects() const {
        return _workRects;
    }
    const Size offset() const {
        return Size(0, 0);
    }
private:
    std::array<Rect, 7> _workRects;
};

struct BordersSize {
    BordersSize(int _left, int _top, int _right, int _bottom) 
        : left(_left), right(_right), top(_top), bottom(_bottom)
    {}
    BordersSize(int w, int h) 
        : left(w), right(w), top(h), bottom(h)
    {}
    BordersSize()
        : BordersSize(0, 0, 0, 0)
    {}
    int left;
    int top;
    int right;
    int bottom;
};

class OpLayoutsServer {
public:

    /*_______________________
      |bbbbbbb border bbbbbb|
      |bbbbbbbbbbbbbbbbbbbbb|
      |bb gpu gpu gpu gpu  b|
      |bb gpu (prime) gpu  b|
      |bb gpu gpu gpu gpu  b|
      |bb gpu gpu gpu gpu  b|
      |bbb cache margin bbbb|
      |bb cpu cpu  cpu cpu b|
      |bb cpu (second) cpu b|
      |bb cpu cpu  cpu cpu b|
      -----------------------*/

    OpLayoutsServer() {}

    /* Returns false if params unsupported */
    bool accept(const WorkSize& workSize, float primeWorkPart,
                const WorkerDescriptor& prime, const WorkerDescriptor& scnd = WorkerDescriptor(Size(0, 0), true), const WorkerDescriptor& border = WorkerDescriptor(Size(0, 0), true),
                const BordersSize& bordersSize = BordersSize(), bool cacheSeparated = false) {
        
        if (primeWorkPart < 1.f && !(scnd.bytesPerIteration.width * scnd.bytesPerIteration.height))
            return false;
        
        const int bordersHeight = (bordersSize.top + bordersSize.bottom);
        const int bordersWidth = (bordersSize.left + bordersSize.right);

        if (primeWorkPart < 0.f || primeWorkPart > 1.f)
            return false;
        if (workSize.height <= bordersHeight || workSize.width <= bordersWidth)
            return false;
        //if (primeWorkPart * workSize.height <= bordersHeight)
        //    return false;
        //if (!prime.considersBorders && primeWorkPart != 1.f && (workSize.height - workSize.height * primeWorkPart <= bordersHeight))
        //    return accept(workSize, 1.f, prime, scnd, border, borderLeftTopSize, borderRightBotSize, cacheSeparated);

        Size primeWorkSize;
        primeWorkSize.width = workSize.width;
        primeWorkSize.height = workSize.height * primeWorkPart;

        Rect primeSpace;

        if (prime.considersBorders) {
            primeSpace.x = 0;
            primeSpace.y = 0;
            primeSpace.width = primeWorkSize.width;
            primeSpace.height = primeWorkSize.height;
        } else {
            primeSpace.x = bordersSize.left;
            primeSpace.y = bordersSize.top;
            primeSpace.width = ((primeWorkSize.width - bordersWidth) / prime.bytesPerIteration.width) * prime.bytesPerIteration.width;
            primeSpace.height = ((primeWorkSize.height - bordersHeight) / prime.bytesPerIteration.height) * prime.bytesPerIteration.height;
        }
        c_stdout << "primeSpace " << primeSpace << endl;
        Point primeStart;
        Point primeEnd;

        primeStart.x = primeSpace.x;
        primeStart.y = primeSpace.y;

        primeEnd.x = primeSpace.x + primeSpace.width;
        primeEnd.y = primeSpace.y + primeSpace.height;
        
        size_t borderRowSizeBytes = border.bytesPerIteration.height * workSize.width;

        const size_t cacheLineSize = 64;
        size_t bottomCacheLineIsolationMargin = (!borderRowSizeBytes || !primeSpace.height || primeWorkPart == 1.f || !cacheSeparated) ?
            0 :
            div(cacheLineSize, borderRowSizeBytes, true);
        c_stdout << "bottomCacheLineIsolationMargin " << bottomCacheLineIsolationMargin << endl;

        if (primeEnd.y + bottomCacheLineIsolationMargin > workSize.height - ((primeWorkPart == 1.f) ? 0 : bordersSize.bottom)) {
            if (primeWorkPart == 1.f)
                c_stdout << "bad" << endl;
            c_stdout << "rec" << endl;
            return accept(workSize, 1.f, prime, scnd, border, bordersSize, cacheSeparated);
        }

        Rect secondSpace;
        if (primeWorkPart < 1.f) {
            if (scnd.considersBorders) {
                secondSpace.x = 0;
                secondSpace.y = primeEnd.y + bottomCacheLineIsolationMargin;
                secondSpace.width = workSize.width;
                secondSpace.height = workSize.height - secondSpace.y;
            } else {
                secondSpace.x = bordersSize.left;
                secondSpace.y = max(primeEnd.y + bottomCacheLineIsolationMargin, (size_t)bordersSize.bottom);//bordersSize.bottom;
                secondSpace.width = ((workSize.width - secondSpace.x) / scnd.bytesPerIteration.width) * scnd.bytesPerIteration.width;
                secondSpace.height = ((workSize.height - secondSpace.y) / scnd.bytesPerIteration.height) * scnd.bytesPerIteration.height;
            }
        }
        c_stdout << "secondSpace " << secondSpace << endl;

        _primeLayout = SolidOperationLayout(prime, primeSpace);
        _secondLayout = SolidOperationLayout(scnd, secondSpace);
        _borderLayout = BorderOperationLayout(border, workSize, primeSpace, secondSpace);

        return true;
    }

    bool accept(const WorkSize& workSize,
                const WorkerDescriptor& prime, const WorkerDescriptor& border = WorkerDescriptor(Size(0, 0), true),
                const BordersSize& bordersSize = BordersSize()) {
        return accept(workSize, 1.f, prime, WorkerDescriptor(Size(0, 0), true), border, bordersSize, false);
    }
    
    using PrimeLayoutT = SolidOperationLayout;
    const PrimeLayoutT& primeLayout() {
        return _primeLayout;
    }
    using SecondLayoutT = SolidOperationLayout;
    const SecondLayoutT& secondLayout() {
        return _secondLayout;
    }
    using BorderLayoutT = BorderOperationLayout;
    const BorderLayoutT& borderLayout() {
        return _borderLayout;
    }

private:

    SolidOperationLayout _primeLayout;
    SolidOperationLayout _secondLayout;
    BorderOperationLayout _borderLayout;
};


class Offload2dBuffer {
public:
    Offload2dBuffer() : autoShare(true), shared(false) {}
    ~Offload2dBuffer() {
        unshare();
    }

    Offload2dBuffer(const Offload2dBuffer& b) {
        buffer = b.buffer;
        step = b.step;
        memoryStart = b.memoryStart;
        wholeSizeBytes = b.wholeSizeBytes;
        stepBytes = b.stepBytes;
        autoShare = false;
        shared = b.shared;
    }

    void* buffer;
    size_t step;
    size_t stepBytes;

    void* memoryStart;
    size_t wholeSizeBytes;

    bool autoShare;

    void share() {
                //std::cerr << "_GFX_share " << shared << " " << memoryStart << std::endl;
        if (autoShare && !shared)
	        if (_GFX_share(memoryStart, wholeSizeBytes))
                std::cerr << "_GFX_share failed" << std::endl;
            else
                shared = true;
    }
    void unshare() {
                //std::cerr << "_GFX_unshare " << shared << " " << memoryStart << std::endl;
        if (autoShare && shared)
	        if (_GFX_unshare(memoryStart))
                std::cerr << "_GFX_unshare failed" << std::endl;
            else
                shared = false;
    }
private:
    bool shared;
};


class Adviser {
public:
    using SolidOperationLayoutCpu = SolidOperationLayout;
    const SolidOperationLayoutCpu& cpuLayout() {
        return layout.secondLayout();
    }

    using BorderOperationLayout = BorderOperationLayout;
    const BorderOperationLayout& borderLayout() {
        return layout.borderLayout();
    }

protected:
    Adviser() {};
    OpLayoutsServer layout;
};

template <class AdviserT, class GpuWorkerT, class CpuWorkerT, class MatT, typename... Args>
static bool performHeteroTask(AdviserT& adviser, GpuWorkerT& gpuWorker, const CpuWorkerT& cpuWorker, const MatT& src, MatT& dst, Args... taskArgs) {
    tstart
        
    t2start
    if (!adviser.accept(src, dst, taskArgs...))
        return false;
    t2end("    adviser             ")

    const auto& gpuTask = adviser.advisedGpu();
    
    if (!gpuTask.async) {

        for (auto& roi : adviser.borderLayout().workRects()) {
            //c_stdout << "b cpu " << roi << endl;
            //c_stdout << "width " << src.cols << " height " << src.rows << endl;
            //c_stdout << "width " << dst.cols << " height " << dst.rows << endl;
            if (roi.width && roi.height) 
                cpuWorker(src(roi), dst(roi), taskArgs...);
        }
    t2end("    borders             ")
        
        int k = gpuWorker.start(gpuTask);
        //if (k == 0) {
        //    c_stdout << "k == 0 " << endl;
        //    return false;
        //}
        
    t2end("    start             ")
        
        for (auto& roi : adviser.cpuLayout().workRects()) {
            //c_stdout << "c cpu " << roi << endl;
            if (roi.width && roi.height) 
                cpuWorker(src(roi), dst(roi), taskArgs...);
        }
    t2end("    cpu             ")

        gpuWorker.finalize(k);
    t2end("    wait             ")
        
    tend("    ALL             ")
    tflush
        return true;
    }

    return false;
}

template< class MatT >
static bool makeMatSquare(MatT& mat) {
    if (abs(fmod(sqrt(mat.total() * mat.channels()), 1)) < 0.0001) {
        int sizes[2];
        sizes[0] = sizes[1] = round(sqrt(mat.total() * mat.channels()));
        mat = mat.reshape(1, 2, sizes);
        if (!mat.isContinuous())
            mat = mat.clone();
        return true;
    } else {
        std::c_stdout << "some error at " << __LINE__ << " in " << __FILE__ << std::endl;
        return false;
    }
}

template< class MatT >
static bool makeMatLine(MatT& mat) {
        int sizes[2];
        sizes[0] = 1;
        sizes[1] = mat.total() * mat.channels();
        mat = mat.reshape(1, 2, sizes);
        if (!mat.isContinuous())
            mat = mat.clone();
        return true;
}

template< class MatT >
static void testContin(MatT& mat) {
    if (!mat.isContinuous())
        c_stdout << "!mat.isContinious()" << endl;
}

template <typename T>
static bool inRange(T a, T mi, T ma) {
    return a >= mi && a <= ma;
}

static void splitItersX(Rect iters, std::vector<Rect>& res) {
    if (iters.width * iters.height <= 20000)
        res.push_back(iters);
    else {
        const int parts = min(div(iters.width * iters.height, 10000, true), iters.width);

        const int part = div(iters.width, parts, true);

        for (int i = 0; i < parts - 1; i++) {
            Rect r = iters;
            r.width = part;
            r.x += i * part;

            res.push_back(r);
        }
        Rect r = iters;
        r.width = iters.width - (parts - 1) * part;
        r.x += (parts - 1) * part;

        res.push_back(r);
    }
}

static void splitItersY(Rect iters, std::vector<Rect>& res) {
    if (iters.width * iters.height <= 20000)
        res.push_back(iters);
    else {
        const int parts = min(div(iters.width * iters.height, 10000, true), iters.height);

        const int part = div(iters.height, parts, true);

        for (int i = 0; i < parts - 1; i++) {
            Rect r = iters;
            r.height = part;
            r.y += i * part;

            res.push_back(r);
        }
        Rect r = iters;
        r.height = iters.height - (parts - 1) * part;
        r.y += (parts - 1) * part;

        res.push_back(r);
    }
}

template<typename VecT, typename F>
static void mapSet(const VecT& v, VecT& res, const F& f) {
    for (auto e : v) {
        VecT tmp;
        f(e, tmp);
        for (auto e_ : tmp)
            res.push_back(e_);
    }
}

static void splitIters(Rect iters, std::vector<Rect>& res) {
    std::vector<Rect> resY;
    splitItersY(iters, resY);
    mapSet(resY, res, splitItersX);
}

template <typename RectsT, typename BuffersT>
static std::vector<Rect> prepare(const RectsT& rects, BuffersT& buffers, Size offset) {
    //std::vector<Rect> res;

    std::vector<Rect> tmp;
    tmp.reserve(rects.size());

    for (auto& r : rects) {
        //c_stdout << r << endl;
        tmp.push_back(r);
    }

    //mapSet(tmp, res, &splitIters);
    for (auto& b : buffers) {
        b->buffer = (char*)b->buffer + offset.width + offset.height * b->stepBytes;
        //c_stdout << "offset.width " << offset.width << " offset.height * b->stepBytes " << (offset.height * b->stepBytes) << endl;
    }
    return tmp;//res;
}

static BordersSize clarifyBordersSize(const BordersSize& borders, const Mat& mat, int borderType = 0, int xFac = 1, int yFac = 1) {
    if (borderType & BORDER_ISOLATED)
        return borders;

    Size wholeSize;
    Point ofs;
    mat.locateROI(wholeSize, ofs);

    Rect roi;
    roi.x = ofs.x;
    roi.y = ofs.y;
    roi.width = mat.cols;
    roi.height = mat.rows;
    
    BordersSize res = borders;

    res.left = max(res.left - xFac * roi.x, 0);
    res.top = max(res.top - yFac * roi.y, 0);
    res.right = max(res.right + xFac * (roi.x + roi.width - wholeSize.width), 0);
    res.bottom = max(res.bottom + yFac * (roi.y + roi.height - wholeSize.height), 0);

    return res;
}

#define te(expr) if (expr) c_stdout << "some error at " << __LINE__ << " in " << __FILE__ << std::endl

#define d_GFX_share(expr1, expr2) te(_GFX_share(expr1, expr2) < 0)
#define d_GFX_unshare(expr) te(_GFX_unshare(expr) < 0)
#define d_GFX_wait(expr) te(_GFX_wait(expr) < 0)

}
}

