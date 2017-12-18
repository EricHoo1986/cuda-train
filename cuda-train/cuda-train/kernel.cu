#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
using namespace std;
#define N 10

void addVector(const int *a, const int *b, int *c, int number) {
	for (int i = 0; i < number; ++i) {
		c[i] = a[i] + b[i];
	}
}

__global__
void addVectorKernel(int *a, int *b, int *c, int number) {
	int tid = blockIdx.x;
	if (tid < number) {
		c[tid] = a[tid] + b[tid];
	}
}

int main()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 100, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	//addVector(a, b, c, 5);

	int size = 5 * sizeof(int);
	int *device_a, *device_b, *device_c;

	cudaMalloc((void **)&device_a, size);
	cudaMalloc((void **)&device_b, size);
	cudaMalloc((void **)&device_c, size);

	cudaMemcpy(device_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, b, size, cudaMemcpyHostToDevice);
	//kernel_fuc<<<blockPerGrid, threadsPerBlock >> >();
	addVectorKernel << <N, 1 >> > (device_a, device_b, device_c, 5);


	cudaMemcpy(c, device_c, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < 5; i++) {
		cout << c[i] << endl;
	}


	// 以下获取显卡属性
	cudaDeviceProp deviceProp;
	int deviceCount;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&deviceCount);
	for (int dev = 0; dev < deviceCount; dev++)
	{
		int driver_version{ 0 }, runtime_version{ 0 };
		cudaDeviceProp device_prop;
		cudaSetDevice(dev);
		/* cudaGetDeviceProperties: 获取指定的GPU设备属性相关信息 */
		cudaGetDeviceProperties(&device_prop, dev);

		fprintf(stdout, "\n设备 %d 名字: %s\n", dev, device_prop.name);

		/* cudaDriverGetVersion: 获取CUDA驱动版本 */
		cudaDriverGetVersion(&driver_version);
		fprintf(stdout, "CUDA驱动版本： %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
		/* cudaRuntimeGetVersion: 获取CUDA运行时版本 */
		cudaRuntimeGetVersion(&runtime_version);
		fprintf(stdout, "CUDA运行时版本： %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);

		fprintf(stdout, "设备计算能力： %d.%d\n", device_prop.major, device_prop.minor);
		fprintf(stdout, "设备上可用的全局内存总量： %f MB, %llu bytes\n",
			(float)device_prop.totalGlobalMem / (1024 * 1024), (unsigned long long)device_prop.totalGlobalMem);
		fprintf(stdout, "每一个线程块上可用的共享内存总量： %f KB, %lu bytes\n",
			(float)device_prop.sharedMemPerBlock / 1024, device_prop.sharedMemPerBlock);
		fprintf(stdout, "每一个线程块上可用的32位寄存器数量: %d\n", device_prop.regsPerBlock);
		fprintf(stdout, "一个线程束包含的线程数量： %d\n", device_prop.warpSize);
		fprintf(stdout, "在内存拷贝中允许的最大pitch数: %d bytes\n", device_prop.memPitch);
		fprintf(stdout, "每一个线程块中支持的最大线程数量: %d\n", device_prop.maxThreadsPerBlock);
		fprintf(stdout, "每一个线程块的每个维度的最大大小(x,y,z): (%d, %d, %d)\n",
			device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
		fprintf(stdout, "每一个线程格的每个维度的最大大小(x,y,z): (%d, %d, %d)\n",
			device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
		fprintf(stdout, "GPU最大时钟频率: %.0f MHz (%0.2f GHz)\n",
			device_prop.clockRate*1e-3f, device_prop.clockRate*1e-6f);
		fprintf(stdout, "设备上可用的常量内存总量: %lu bytes\n", device_prop.totalConstMem);
		fprintf(stdout, "纹理对齐要求: %lu bytes\n", device_prop.textureAlignment);
		fprintf(stdout, "是否支持设备重叠功能: %s\n", device_prop.deviceOverlap ? "Yes" : "No");
		fprintf(stdout, "设备上多处理器的数量: %d\n", device_prop.multiProcessorCount);
		fprintf(stdout, "执行核函数时是否有运行时间限制: %s\n", device_prop.kernelExecTimeoutEnabled ? "Yes" : "No");
		fprintf(stdout, "设备是否是一个集成GPU: %s\n", device_prop.integrated ? "Yes" : "No");
		fprintf(stdout, "设备是否支持映射主机内存: %s\n", device_prop.canMapHostMemory ? "Yes" : "No");
		fprintf(stdout, "CUDA设备计算模式: %d\n", device_prop.computeMode);
		fprintf(stdout, "一维纹理支持的最大大小: %d\n", device_prop.maxTexture1D);
		fprintf(stdout, "二维纹理支持的最大大小(x,y): (%d, %d)\n", device_prop.maxTexture2D[0], device_prop.maxSurface2D[1]);
		fprintf(stdout, "三维纹理支持的最大大小(x,y,z): (%d, %d, %d)\n",
			device_prop.maxTexture3D[0], device_prop.maxSurface3D[1], device_prop.maxSurface3D[2]);
		fprintf(stdout, "内存时钟频率峰值: %.0f Mhz\n", device_prop.memoryClockRate * 1e-3f);
		fprintf(stdout, "全局内存总线宽度: %d bits\n", device_prop.memoryBusWidth);
		fprintf(stdout, "L2缓存大小: %d bytes\n", device_prop.l2CacheSize);
		fprintf(stdout, "每个多处理器支持的最大线程数量: %d\n", device_prop.maxThreadsPerMultiProcessor);
		fprintf(stdout, "设备是否支持同时执行多个核函数: %s\n", device_prop.concurrentKernels ? "Yes" : "No");
		fprintf(stdout, "异步引擎数量: %d\n", device_prop.asyncEngineCount);
		fprintf(stdout, "是否支持设备与主机共享一个统一的地址空间: %s\n", device_prop.unifiedAddressing ? "Yes" : "No");
	}



	getchar();
	return 0;
}
