@echo off
echo [OpenCV build script]
::Use this string for adding array supporting in batch interpreter
setlocal EnableDelayedExpansion

set build_type[0]=-G "Visual Studio 12 2013 Win64"
set build_type[1]=-G "Visual Studio 12 2013"
set opencv_dir="opencv"
set build=0

set first_arg=%~1
set second_arg=%~2

if defined first_arg (
	if "%first_arg%"=="x86" (
		echo x86 build selected
		set build=1
	) else if "%first_arg%"=="x64" (
		echo x64 build selected
		set build=0
	) else (	
		echo First argument incorrect. Setted x64 by default
	)
) else (
	echo First argument is not provided. Setted x64 by default
)
	
if defined second_arg (
	set opencv_dir=%2
) else (
	echo Second argument is not provided. OpenCV directory setted by default
)

where /Q MSBuild.exe

if ERRORLEVEL 1 (
	echo Can't found MSBuild directory. Add path to MSbuild directory to PATH and try again!
	goto endbuild
)

where /Q ld.exe 

if ERRORLEVEL 1 (
	echo Can't found BinUtils directory. Add path to BinUtils directory to PATH and try again!
	goto endbuild
)

mkdir Build
cd Build

mkdir ipp_build
cd ipp_build
echo CMake build "IPP build"
set currentbuild="ipp"
cmake -DWITH_IPP=ON -DWITH_OPENCL=OFF -DCMAKE_GENERATOR_TOOLSET="Intel C++ Compiler 16.0" !build_type[%build%]! ../../%opencv_dir%
goto compilation
:"ipp" 
cd ..

mkdir opencl_build
cd opencl_build	
echo CMake build "OpenCL build"
set currentbuild="ocl"
cmake -DWITH_OPENCL=ON -DWITH_IPP=OFF -DCMAKE_GENERATOR_TOOLSET="Intel C++ Compiler 16.0" !build_type[%build%]! ../../%opencv_dir%
goto compilation
:"ocl"
cd ..

mkdir igpu_with_cpu
cd igpu_with_cpu
echo CMake build "Intel GFX Offload build" 
set currentbuild="igpu"
cmake -DWITH_OPENCL=OFF -DWITH_IPP=OFF -DWITH_IGPU=ON -DCMAKE_GENERATOR_TOOLSET="Intel C++ Compiler 16.0" !build_type[%build%]! ../../%opencv_dir%
goto compilation
:"igpu"
cd ../../..
goto endbuild

:compilation
echo MSBuild Running...
msbuild ALL_BUILD.vcxproj /p:Configuration=Release
goto %currentbuild%

:endbuild