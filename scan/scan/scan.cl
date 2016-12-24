#define SWAP(a,b) {__local int * tmp=a; a=b; b=tmp;}

__kernel void scan_hillis_steele(__global float * input, __global float * output, __global float * block_sums, __local float * a, __local float * b)
{
    uint lid = get_local_id(0);
	uint group_id = get_group_id(0);
    uint block_size = get_local_size(0);
	uint i = lid + group_id * block_size;
    a[lid] = b[lid] = input[i];
    barrier(CLK_LOCAL_MEM_FENCE);
 
    for(uint s = 1; s < block_size; s <<= 1)
    {
        if(lid > (s-1))
        {
            b[lid] = a[lid] + a[lid-s];
        }
        else
        {
            b[lid] = a[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(a,b);
    }
    output[i] = a[lid];
	if (lid == block_size - 1) {
		block_sums[group_id] = a[lid];
	}
}

__kernel void block_adder(__global float * output, __global float * block_sums) {
	uint lid = get_local_id(0);
	uint group_id = get_group_id(0);
	uint block_size = get_local_size(0);
	uint i = lid + group_id * block_size;
	if (group_id > 0) {
		output[i] += block_sums[group_id - 1];
	}
}
