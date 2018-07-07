#include "cuda_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cassert>
using namespace std;

#define CUDA_CALL( expr ) \
{ \
	auto status=expr; \
	if ( status != cudaSuccess )  \
	{ \
		cout << "CUDA ERROR: " << (status) << endl; \
		assert( 0 ); \
		exit( 1 ); \
	} \
}


extern void start_kernel(int* dev[3], int arraySize);

template <typename T>
void s_p(const string& name, T val)
{
    cout << name.c_str() << " " << val << endl;
}


int main()
{
    cout << "Enter array size" << endl;
    auto arraySize = 1;
    cin >> arraySize;
    cout << "Chosen size: " << arraySize << endl;

    int* host[3];
    int* dev[3];
    cudaError err;

    cudaDeviceProp props{};
    CUDA_CALL(cudaGetDeviceProperties(&props, 0));

    s_p("Name", props.name);
    s_p("GlobalMem", props.totalGlobalMem);
    s_p("SharedMemPerBlock", props.sharedMemPerBlock);

    for ( auto& h : host ) h = new int[arraySize];
    for ( auto& d : dev ) CUDA_CALL(cudaMalloc(&d, arraySize * sizeof(int)));

    const int k = 5;
    for ( int i = 0; i < arraySize; i++ )
    {
        host[0][i] = k + i;
        host[1][i] = k + i;
        host[2][i] = 0;
    }

    for ( int i = 0; i < 3; i++ )
        cudaMemcpy(dev[i], host[i], sizeof(int) * arraySize, cudaMemcpyHostToDevice);

    start_kernel(dev, arraySize);

    cudaMemcpy(host[2], dev[2], sizeof(int) * arraySize, cudaMemcpyDeviceToHost);

    cout << "Result" << endl;
    for ( int i = 0; i < arraySize; i++ )
    {
        cout << "h0[" << i << "]=" << host[0][i] << " ";
        cout << "h1[" << i << "]=" << host[1][i] << " ";
        cout << "h2[" << i << "]=" << host[2][i] << " ";
        cout << endl;
    }
    cout << endl << endl;


    for ( auto h : host ) delete[] h;
    for ( auto d : dev )
        CUDA_CALL(cudaFree(d));

    system("pause");
}
