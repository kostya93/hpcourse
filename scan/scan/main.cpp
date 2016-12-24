#pragma warning(disable : 4996)
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <assert.h>

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
      std::ifstream cl_file("scan.cl");
      std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
         cl_string.length() + 1));

      // create program
      cl::Program program(context, source);

      // compile opencl source
	  size_t const block_size = 256;
      program.build(devices, "-D BLOCK_SIZE=256");

	  std::ifstream in("input.txt");
	  int N;
	  in >> N;
	  int NN = (N % block_size) ? (N / block_size + 1) * block_size : N;
	  int num_of_blocks = NN / block_size;

      // create a message to send to kernel
      
      std::vector<float> input(NN, 0.0);
      std::vector<float> output(NN, 0.0);
	  std::vector<float> blocks_sums(num_of_blocks, 0.0);
      for (size_t i = 0; i < N; ++i)
      {
         in >> input[i];
      }

      // allocate device buffer to hold message
      cl::Buffer dev_input (context, CL_MEM_READ_ONLY, sizeof(float) * NN);
      cl::Buffer dev_output(context, CL_MEM_READ_WRITE, sizeof(float) * NN);
	  cl::Buffer dev_blocks_sums(context, CL_MEM_READ_WRITE, sizeof(float) * num_of_blocks);

      // copy from cpu to gpu
      queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * NN, &input[0]);

      // load named kernel from opencl source
      cl::Kernel kernel_hs(program, "scan_hillis_steele");
      cl::KernelFunctor scan_hs(kernel_hs, queue, cl::NullRange, cl::NDRange(NN), cl::NDRange(block_size));
	  scan_hs(dev_input, dev_output, dev_blocks_sums, cl::__local(sizeof(float) * NN), cl::__local(sizeof(float) * NN));

	  if (num_of_blocks > 1) {
		  cl::Kernel kernel_block_adder(program, "block_adder");
		  cl::KernelFunctor block_adder(kernel_block_adder, queue, cl::NullRange, cl::NDRange(NN), cl::NDRange(block_size));
		  block_adder(dev_output, dev_blocks_sums);
	  }

	  queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * NN, &output[0]);

	  std::ofstream out("output.txt");
	  out << std::setprecision(2) << std::fixed;
	  for (int i = 0; i < N; i++) {
		  out << output[i] << " ";
	  }
	  out << std::endl;

   }
   catch (cl::Error e)
   {
      std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
   }

   return 0;
}
