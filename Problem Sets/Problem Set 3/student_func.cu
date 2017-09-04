/* Udacity Homework 3
HDR Tone-mapping

Background HDR
==============

A High Dynamic Range (HDR) image contains a wider variation of intensity
and color than is allowed by the RGB format with 1 byte per channel that we
have used in the previous assignment.

To store this extra information we use single precision floating point for
each channel.  This allows for an extremely wide range of intensity values.

In the image for this assignment, the inside of church with light coming in
through stained glass windows, the raw input floating point values for the
channels range from 0 to 275.  But the mean is .41 and 98% of the values are
less than 3!  This means that certain areas (the windows) are extremely bright
compared to everywhere else.  If we linearly map this [0-275] range into the
[0-255] range that we have been using then most values will be mapped to zero!
The only thing we will be able to see are the very brightest areas - the
windows - everything else will appear pitch black.

The problem is that although we have cameras capable of recording the wide
range of intensity that exists in the real world our monitors are not capable
of displaying them.  Our eyes are also quite capable of observing a much wider
range of intensities than our image formats / monitors are capable of
displaying.

Tone-mapping is a process that transforms the intensities in the image so that
the brightest values aren't nearly so far away from the mean.  That way when
we transform the values into [0-255] we can actually see the entire image.
There are many ways to perform this process and it is as much an art as a
science - there is no single "right" answer.  In this homework we will
implement one possible technique.

Background Chrominance-Luminance
================================

The RGB space that we have been using to represent images can be thought of as
one possible set of axes spanning a three dimensional space of color.  We
sometimes choose other axes to represent this space because they make certain
operations more convenient.

Another possible way of representing a color image is to separate the color
information (chromaticity) from the brightness information.  There are
multiple different methods for doing this - a common one during the analog
television days was known as Chrominance-Luminance or YUV.

We choose to represent the image in this way so that we can remap only the
intensity channel and then recombine the new intensity values with the color
information to form the final image.

Old TV signals used to be transmitted in this way so that black & white
televisions could display the luminance channel while color televisions would
display all three of the channels.


Tone-mapping
============

In this assignment we are going to transform the luminance channel (actually
the log of the luminance, but this is unimportant for the parts of the
algorithm that you will be implementing) by compressing its range to [0, 1].
To do this we need the cumulative distribution of the luminance values.

Example
-------

input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
min / max / range: 0 / 9 / 9

histo with 3 bins: [4 7 3]

cdf : [4 11 14]


Your task is to calculate this cumulative distribution by following these
steps.

*/


#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>

//for this algorithm, it is important that the block size is a power of two
//or, due to integer divison, some threads will not get reduced and be missed the second time
//see https://discussions.udacity.com/t/problem-set-3-hint-about-min-and-max-values-of-luminance-channel/224268/3

__global__ void reduceMin(const float* d_in, float* d_out)
{
	int abs_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_x = threadIdx.x;

	extern __shared__ float sdata[];

	sdata[thread_x] = d_in[abs_x];
	__syncthreads();

	for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1)
	{
		//works for now, I assume it's a consequence of how this data is entered
		if (thread_x < i)
		{
			sdata[thread_x] = min(sdata[thread_x], sdata[thread_x + i]);
		}

		__syncthreads();
	}

	//return result at the 0th thread of every block:
	if (thread_x == 0)
	{
		d_out[blockIdx.x] = sdata[0];
	}
}

__global__ void reduceMax(const float* d_in, float* d_out)
{
	int abs_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_x = threadIdx.x;

	extern __shared__ float sdata[];

	sdata[thread_x] = d_in[abs_x];
	__syncthreads();

	int last_i = blockDim.x;
	for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1)
	{
		if (thread_x < i)
		{
			sdata[thread_x] = max(sdata[thread_x], sdata[thread_x + i]);

			//this checks for weird edge cases where the block dimension is not a power of two
			//see https://discussions.udacity.com/t/wrong-max-value-problem-set-3/85232/7

			//basically, if we are at the "last" thread of this iteration (i - 1)
			//and if we lost a point due to integer divison
			if (thread_x == i - 1 && last_i > 2 * i)
			{
				//then take the point we lost to integer divison at (last_i - 1)
				sdata[thread_x] = max(sdata[thread_x], sdata[last_i - 1]);
			}
		}

		__syncthreads();
		last_i = i;
	}

	//return result at the 0th thread of every block:
	if (thread_x == 0)
	{
		d_out[blockIdx.x] = sdata[0];
	}
}

void reduce(const float* d_in, float* d_out, size_t size, bool type)
{
	float* d_intermediate;
	checkCudaErrors(cudaMalloc((void **)&d_intermediate, sizeof(float) * size));

	int threads = 1024;
	int blocks = (size - 1) / threads + 1;

	if (type) //meaning reduceMin
	{
		reduceMin<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_intermediate);

		threads = blocks;
		blocks = 1;

		reduceMin<<<blocks, threads, threads * sizeof(float)>>>(d_intermediate, d_out);
	}
	else
	{
		reduceMax<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_intermediate);

		threads = blocks;
		blocks = 1;

		reduceMax<<<blocks, threads, threads * sizeof(float)>>>(d_intermediate, d_out);
	}

	checkCudaErrors(cudaFree(d_intermediate));
}

__global__ void histogram(const float* d_in, unsigned int* d_out,
	const float lumMin, const float lumRange, const size_t numBins, const size_t size)
{
	int abs_x = threadIdx.x + blockDim.x * blockIdx.x;

	if (abs_x > size)
	{
		return;
	}

	int bin = (d_in[abs_x] - lumMin) / lumRange * numBins;

	//then increment:
	atomicAdd(&(d_out[bin]), 1);
}

__global__ void inclusivePrefixAdd(unsigned int* d_in, unsigned int* d_out)
{
	//Hillis Steele implementation
	//NOTE: right now, this is only set up for 1 block of 1024 threads

	int abs_x = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_x = threadIdx.x;

	extern __shared__ unsigned int segment[];
	segment[thread_x] = d_in[abs_x];
	//d_out[thread_x] = d_in[thread_x];
	__syncthreads();

	for (unsigned int i = 1; i < blockDim.x; i <<= 1)
	{
		if (thread_x >= i)
		{
			//d_out[thread_x] = d_out[thread_x] + d_out[thread_x - i];
			segment[thread_x] = segment[thread_x] + segment[thread_x - i];
		}

		__syncthreads();
	}

	//this happens in different blocks, so no need to syncthreads()
	if (blockIdx.x > 0)
	{
		//carry over the result of the last segment
		segment[thread_x] = segment[thread_x] + d_out[blockDim.x * (blockIdx.x - 1)];
	}

	d_out[abs_x] = segment[thread_x];
}

__global__ void exclusivePrefixAdd(unsigned int* d_in, unsigned int* d_out)
{
	//Belloch implementation
	//NOTE: this is set up specifically for 1 block of 1024 threads

	int thread_x = threadIdx.x;

	d_out[thread_x] = d_in[thread_x];
	__syncthreads();

	//first, do the reduce:
	for (unsigned int i = 2; i <= blockDim.x; i <<= 1)
	{
		if ((thread_x + 1) % i == 0)
		{
			d_out[thread_x] = d_out[thread_x] + d_out[thread_x - i / 2];
		}

		__syncthreads();
	}


	//now do the downsweep part:

	if (thread_x == blockDim.x - 1)
	{
		d_out[thread_x] = 0;
	}

	//maybe need a syncthreads() here because of that write above? it's only 1 thread so idk if it affects it

	for (unsigned int i = blockDim.x; i >= 2; i >>= 1)
	{
		if ((thread_x + 1) % i == 0)
		{
			unsigned int temp = d_out[thread_x - (i / 2)];

			//the "left" copy
			d_out[thread_x - (i / 2)] = d_out[thread_x];

			//and the "right" operation
			d_out[thread_x] = temp + d_out[thread_x];
		}
		__syncthreads();
	}

}

void show(unsigned int* d_in, int size)
{
	unsigned int * temp;
	temp = (unsigned int *)malloc(sizeof(unsigned int) * size);

	checkCudaErrors(cudaMemcpy(temp, d_in, sizeof(unsigned int) * size, cudaMemcpyDeviceToHost));

	for (int i = 0; i < size; i++)
	{
		printf("Bin: %d is %d\n", i, temp[i]);
	}

	free(temp);
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
	unsigned int* const d_cdf,
	float &min_logLum,
	float &max_logLum,
	const size_t numRows,
	const size_t numCols,
	const size_t numBins)
{
	//TODO
	/*Here are the steps you need to implement
	1) find the minimum and maximum value in the input logLuminance channel
	store in min_logLum and max_logLum
	2) subtract them to find the range
	3) generate a histogram of all the values in the logLuminance channel using
	the formula: bin = (lum[i] - lumMin) / lumRange * numBins
	4) Perform an exclusive scan (prefix sum) on the histogram to get
	the cumulative distribution of luminance values (this should go in the
	incoming d_cdf pointer which already has been allocated for you)       */

	//memory to hold result of reduce:
	float* d_out;
	checkCudaErrors(cudaMalloc((void **)&d_out, sizeof(float)));

	// --- 1) MIN AND MAX ---

	//first, find min:
	reduce(d_logLuminance, d_out, numRows * numCols, true);
	checkCudaErrors(cudaMemcpy(&min_logLum, d_out, sizeof(float), cudaMemcpyDeviceToHost));

	//now max:
	reduce(d_logLuminance, d_out, numRows * numCols, false);
	checkCudaErrors(cudaMemcpy(&max_logLum, d_out, sizeof(float), cudaMemcpyDeviceToHost));

	// --- 2) RANGE ---
	float lumRange = max_logLum - min_logLum;

	//--- 3) HISTOGRAM ---

	//create the bins and initialize
	unsigned int* h_bins;
	h_bins = (unsigned int *)malloc(sizeof(unsigned int) * numBins);

	for (int i = 0; i < numBins; i++)
	{
		h_bins[i] = 0;
	}

	unsigned int* d_bins;
	checkCudaErrors(cudaMalloc((void **)&d_bins, sizeof(unsigned int) * numBins));

	checkCudaErrors(cudaMemcpy(d_bins, h_bins, sizeof(unsigned int) * numBins, cudaMemcpyHostToDevice));

	int threads = 1024;
	int blocks = (numRows * numCols - 1) / threads + 1;
	histogram<<<blocks, threads>>>(d_logLuminance, d_bins, min_logLum, lumRange, numBins, numCols * numRows);

	//use the result in d_bins for the sum

	//--- 4) EXCLUSIVE PREFIX SUM ---

	//sometimes this works and I don't know why
	//inclusivePrefixAdd<<<1, 1024, sizeof(unsigned int) * 1024>>>(d_bins, d_cdf);

	//but this works and that's what should happen
	exclusivePrefixAdd<<<1, 1024>>>(d_bins, d_cdf);

	free(h_bins);
	checkCudaErrors(cudaFree(d_out));
	checkCudaErrors(cudaFree(d_bins));
}