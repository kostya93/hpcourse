#pragma warning(disable : 4996)
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>

int main()
{
   std::vector<cl::Platform> platforms;
   std::vector<cl::Device> devices;
   std::vector<cl::Kernel> kernels;

   try {

      // create platform
      cl::Platform::get(&platforms);
      platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

      // create context
      cl::Context context(devices);

      // create command queue
      cl::CommandQueue queue(context, devices[0]);

      // load opencl source
      std::ifstream cl_file("matrix_mult.cl");
      std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
         cl_string.length() + 1));

      // create program
      cl::Program program(context, source);

      // compile opencl source
	  size_t const block_size = 16;
      program.build(devices, "-D BLOCK_SIZE=16");
	
	  std::ifstream input("input.txt");
	  const int MAX_SIZE = 1024;

	  int N, M;
	  input >> N >> M;

	  int size_A = N * N;
	  int size_B = M * M;
	  int size_C = size_A;

	  std::vector<float> A(size_A, 0);
	  std::vector<float> B(size_B, 0);
	  std::vector<float> C(size_C, 0);
	  
      for (size_t i = 0; i < N; ++i)
      {
         for (size_t j = 0; j < N; ++j)
         {
            size_t idx = i * N + j;
			input >> A[idx];
         }
      }

	  for (size_t i = 0; i < M; ++i)
	  {
		  for (size_t j = 0; j < M; ++j)
		  {
			  size_t idx = i * M + j;
			  input >> B[idx];
		  }
	  }

      // allocate device buffer to hold message
      cl::Buffer dev_a(context, CL_MEM_READ_ONLY,  sizeof(float) * size_A);
      cl::Buffer dev_b(context, CL_MEM_READ_ONLY,  sizeof(float) * size_B);
      cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * size_C);

      // copy from cpu to gpu
      queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * size_A, &A[0]);
      queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(float) * size_B, &B[0]);

      // load named kernel from opencl source
	  cl::Kernel kernel(program, "matrix_mult");
	  int sz = (N % block_size) ? (N / block_size + 1) * block_size : N;
	  cl::KernelFunctor matrix_mult(kernel, queue, cl::NullRange, cl::NDRange(sz, sz), cl::NDRange(block_size, block_size));
	  matrix_mult(dev_a, dev_b, dev_c, N, M);

      queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(float) * size_C, &C[0]);

      for (size_t i = 0; i < N; ++i)
      {
         for (size_t j = 0; j < N; ++j)
         {
            size_t idx = i * N + j;
            std::cout << C[idx] << " ";
         }
         std::cout << '\n';
      }
      std::cout << '\n';

   }
   catch (cl::Error e)
   {
      std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
   }

   return 0;
}