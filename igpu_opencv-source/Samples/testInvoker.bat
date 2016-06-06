@echo off
echo [GFX OpenCV test script]
where /Q opencv_core310.dll opencv_imgproc310.dll opencv_imgcodecs310.dll

IF ERRORLEVEL 1 (
	echo Can't found OpenCV *.dll files. Add OpenCV *.dll files to this directory and try again!
	goto test
)

echo [Bilateral] 
cd Bilateral
ffmpeg.exe -i original.avi -y -vf "ocv=smooth:bilateral|11|11" temp.avi
cd ..

echo [Canny]
copy /Y opencv_core310.dll Canny
copy /Y opencv_imgcodecs310.dll Canny
copy /Y opencv_imgproc310.dll Canny
cd Canny
canny_test.exe
del opencv*.dll
cd ..

echo [ColorCanny]
copy /Y opencv_core310.dll ColorCanny
copy /Y opencv_imgcodecs310.dll ColorCanny
copy /Y opencv_imgproc310.dll ColorCanny
cd ColorCanny
colorcanny_test.exe
del opencv*.dll
cd ..

echo [BoxFilter]
copy /Y opencv_core310.dll BoxFilter
copy /Y opencv_imgcodecs310.dll BoxFilter
copy /Y opencv_imgproc310.dll BoxFilter
cd BoxFilter
boxfilter_test.exe
del opencv*.dll
cd ..

echo [GEMM]
copy /Y opencv_core310.dll GEMM
copy /Y opencv_imgcodecs310.dll GEMM
copy /Y opencv_imgproc310.dll GEMM
cd GEMM
gemm_test.exe
del opencv*.dll
cd ..

echo [Filter2D]
copy /Y opencv_core310.dll Filter2D
copy /Y opencv_imgcodecs310.dll Filter2D
copy /Y opencv_imgproc310.dll Filter2D
cd Filter2D
filter2d_test.exe
del opencv*.dll
cd ..


echo [sepFilter2D]
copy /Y opencv_core310.dll sepFilter2D
copy /Y opencv_imgcodecs310.dll sepFilter2D
copy /Y opencv_imgproc310.dll sepFilter2D
cd sepFilter2D
sepfilter2D_test.exe
del opencv*.dll
cd ..

echo [Morph]
copy /Y opencv_core310.dll Morph
copy /Y opencv_imgcodecs310.dll Morph
copy /Y opencv_imgproc310.dll Morph
cd Morph
morph_test.exe
del opencv*.dll
cd ..

:test