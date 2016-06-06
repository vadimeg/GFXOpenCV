@echo off
echo [GFX OpenCV test build script]
where /Q cl.exe

if ERRORLEVEL 1 (
	echo Can't found cl.exe. Add path to cl.exe and other VS environment directory to PATH and try again!
	goto endbuild
)

echo [Building ImageGenerator]
icl /IOpenCV\include\ /EHsc ImageGenerator/imageGenerator.cpp /link /LIBPATH:OpenCV\lib "opencv_imgproc310.lib" "opencv_imgcodecs310.lib" "opencv_core310.lib" /OUT:ImageGenerator/imageGenerator.exe

echo [Building OpenCV Tests]
icl /IOpenCV\include\ /EHsc Canny/canny_test.cpp /link /LIBPATH:OpenCV\lib "opencv_imgproc310.lib" "opencv_imgcodecs310.lib" "opencv_core310.lib" /OUT:Canny/canny_test.exe
icl /IOpenCV\include\ /EHsc ColorCanny/colorcanny_test.cpp /link /LIBPATH:OpenCV\lib "opencv_imgproc310.lib" "opencv_imgcodecs310.lib" "opencv_core310.lib" /OUT:ColorCanny/colorcanny_test.exe
icl /IOpenCV\include\ /EHsc BoxFilter/boxFilter_test.cpp /link /LIBPATH:OpenCV\lib "opencv_imgproc310.lib" "opencv_imgcodecs310.lib" "opencv_core310.lib" /OUT:BoxFilter/boxfilter_test.exe
icl /IOpenCV\include\ /EHsc Filter2D/Filter2D_test.cpp /link /LIBPATH:OpenCV\lib "opencv_imgproc310.lib" "opencv_imgcodecs310.lib" "opencv_core310.lib" /OUT:Filter2D/filter2d_test.exe
icl /IOpenCV\include\ /EHsc Morph/morph_test.cpp /link /LIBPATH:OpenCV\lib "opencv_imgproc310.lib" "opencv_imgcodecs310.lib" "opencv_core310.lib" /OUT:Morph/morph_test.exe
icl /IOpenCV\include\ /EHsc GEMM/gemm_test.cpp /link /LIBPATH:OpenCV\lib "opencv_imgproc310.lib" "opencv_imgcodecs310.lib" "opencv_core310.lib" /OUT:GEMM/gemm_test.exe
icl /IOpenCV\include\ /EHsc sepFilter2D/sepFilter2D_test.cpp /link /LIBPATH:OpenCV\lib "opencv_imgproc310.lib" "opencv_imgcodecs310.lib" "opencv_core310.lib" /OUT:sepFilter2D/sepfilter2d_test.exe
del *.obj
:endbuild


