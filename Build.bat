@echo off
echo [OpenCV Build Batch File]
set currentbuild="build1"
mkdir Build
cd Build

mkdir IPP_Host
cd IPP_Host
echo CMake Build "IPP + Host (CPU)"
cmake -DWITH_IPP=ON -DCMAKE_GENERATOR_TOOLSET="Intel C++ Compiler 16.0" ../..
goto compilation
:"build1" 
cd ..

mkdir OpenCL_CPU
cd OpenCL_CPU	
echo CMake Build "OpenCL + Host (CPU)"
set currentbuild="build2"
cmake -DWITH_OPENCL=ON -DCMAKE_GENERATOR_TOOLSET="Intel C++ Compiler 16.0" ../..
goto compilation
:"build2"
cd ..

mkdir OpenCL_GPU
cd OpenCL_GPU
echo CMake Build "OpenCL + Host (GPU)"
set currentbuild="build3" 
cmake -DWITH_OPENCL=ON -DCMAKE_GENERATOR_TOOLSET="Intel C++ Compiler 16.0" ../..
goto compilation
:"build3"
cd ..

mkdir GFX_Host
cd GFX_Host
echo CMake Build "Intel GFX Offload (GPU)" 
set currentbuild="build4"
cmake -DCMAKE_GENERATOR_TOOLSET="Intel C++ Compiler 16.0" ../..
goto compilation
:"build4"
cd ..

goto endbuild

:compilation
echo MSBuild Running...
if %PROCESSOR_ARCHITECTURE%==x86 (	
"C:\Program Files\MSBuild\12.0\Bin\MSBuild.exe" ALL_BUILD.vcxproj /p:Configuration=Release
goto %currentbuild%)
	
"C:\Program Files (x86)\MSBuild\12.0\Bin\MSBuild.exe" ALL_BUILD.vcxproj /p:Configuration=Release
goto %currentbuild%

:endbuild
cd ../..