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


	// ���»�ȡ�Կ�����
	cudaDeviceProp deviceProp;
	int deviceCount;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&deviceCount);
	for (int dev = 0; dev < deviceCount; dev++)
	{
		int driver_version{ 0 }, runtime_version{ 0 };
		cudaDeviceProp device_prop;
		cudaSetDevice(dev);
		/* cudaGetDeviceProperties: ��ȡָ����GPU�豸���������Ϣ */
		cudaGetDeviceProperties(&device_prop, dev);

		fprintf(stdout, "\n�豸 %d ����: %s\n", dev, device_prop.name);

		/* cudaDriverGetVersion: ��ȡCUDA�����汾 */
		cudaDriverGetVersion(&driver_version);
		fprintf(stdout, "CUDA�����汾�� %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
		/* cudaRuntimeGetVersion: ��ȡCUDA����ʱ�汾 */
		cudaRuntimeGetVersion(&runtime_version);
		fprintf(stdout, "CUDA����ʱ�汾�� %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);

		fprintf(stdout, "�豸���������� %d.%d\n", device_prop.major, device_prop.minor);
		fprintf(stdout, "�豸�Ͽ��õ�ȫ���ڴ������� %f MB, %llu bytes\n",
			(float)device_prop.totalGlobalMem / (1024 * 1024), (unsigned long long)device_prop.totalGlobalMem);
		fprintf(stdout, "ÿһ���߳̿��Ͽ��õĹ����ڴ������� %f KB, %lu bytes\n",
			(float)device_prop.sharedMemPerBlock / 1024, device_prop.sharedMemPerBlock);
		fprintf(stdout, "ÿһ���߳̿��Ͽ��õ�32λ�Ĵ�������: %d\n", device_prop.regsPerBlock);
		fprintf(stdout, "һ���߳����������߳������� %d\n", device_prop.warpSize);
		fprintf(stdout, "���ڴ濽������������pitch��: %d bytes\n", device_prop.memPitch);
		fprintf(stdout, "ÿһ���߳̿���֧�ֵ�����߳�����: %d\n", device_prop.maxThreadsPerBlock);
		fprintf(stdout, "ÿһ���߳̿��ÿ��ά�ȵ�����С(x,y,z): (%d, %d, %d)\n",
			device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
		fprintf(stdout, "ÿһ���̸߳��ÿ��ά�ȵ�����С(x,y,z): (%d, %d, %d)\n",
			device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
		fprintf(stdout, "GPU���ʱ��Ƶ��: %.0f MHz (%0.2f GHz)\n",
			device_prop.clockRate*1e-3f, device_prop.clockRate*1e-6f);
		fprintf(stdout, "�豸�Ͽ��õĳ����ڴ�����: %lu bytes\n", device_prop.totalConstMem);
		fprintf(stdout, "�������Ҫ��: %lu bytes\n", device_prop.textureAlignment);
		fprintf(stdout, "�Ƿ�֧���豸�ص�����: %s\n", device_prop.deviceOverlap ? "Yes" : "No");
		fprintf(stdout, "�豸�϶ദ����������: %d\n", device_prop.multiProcessorCount);
		fprintf(stdout, "ִ�к˺���ʱ�Ƿ�������ʱ������: %s\n", device_prop.kernelExecTimeoutEnabled ? "Yes" : "No");
		fprintf(stdout, "�豸�Ƿ���һ������GPU: %s\n", device_prop.integrated ? "Yes" : "No");
		fprintf(stdout, "�豸�Ƿ�֧��ӳ�������ڴ�: %s\n", device_prop.canMapHostMemory ? "Yes" : "No");
		fprintf(stdout, "CUDA�豸����ģʽ: %d\n", device_prop.computeMode);
		fprintf(stdout, "һά����֧�ֵ�����С: %d\n", device_prop.maxTexture1D);
		fprintf(stdout, "��ά����֧�ֵ�����С(x,y): (%d, %d)\n", device_prop.maxTexture2D[0], device_prop.maxSurface2D[1]);
		fprintf(stdout, "��ά����֧�ֵ�����С(x,y,z): (%d, %d, %d)\n",
			device_prop.maxTexture3D[0], device_prop.maxSurface3D[1], device_prop.maxSurface3D[2]);
		fprintf(stdout, "�ڴ�ʱ��Ƶ�ʷ�ֵ: %.0f Mhz\n", device_prop.memoryClockRate * 1e-3f);
		fprintf(stdout, "ȫ���ڴ����߿��: %d bits\n", device_prop.memoryBusWidth);
		fprintf(stdout, "L2�����С: %d bytes\n", device_prop.l2CacheSize);
		fprintf(stdout, "ÿ���ദ����֧�ֵ�����߳�����: %d\n", device_prop.maxThreadsPerMultiProcessor);
		fprintf(stdout, "�豸�Ƿ�֧��ͬʱִ�ж���˺���: %s\n", device_prop.concurrentKernels ? "Yes" : "No");
		fprintf(stdout, "�첽��������: %d\n", device_prop.asyncEngineCount);
		fprintf(stdout, "�Ƿ�֧���豸����������һ��ͳһ�ĵ�ַ�ռ�: %s\n", device_prop.unifiedAddressing ? "Yes" : "No");
	}



	getchar();
	return 0;
}
