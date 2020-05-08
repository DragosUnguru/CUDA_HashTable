#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#include "gpu_hashtable.hpp"

__global__ void insert_entries(int* keys, int* values, int numKeys, KeyValue* hashtable, unsigned int capacity) {
	unsigned int idx;
	unsigned int key;
	unsigned int value;
	unsigned int hashedKey;
	unsigned int oldVal;

	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numKeys)
		return;

	key = keys[idx];
	value = values[idx];
	hashedKey = hashFunc(key, capacity);

	while (true) {
		oldVal = atomicCAS(&hashtable[hashedKey].key, KEY_INVALID, key);

		if (oldVal == KEY_INVALID || oldVal == key) {
			hashtable[hashedKey].value = value;
			break;
		}

		++hashedKey;
		hashedKey %= capacity;
	}
}

__global__ void get_values(int* keys, int numKeys, KeyValue* hashtable, unsigned int capacity, int* deviceResult) {
	unsigned int idx;
	unsigned int key;
	unsigned int hashedKey;
	unsigned int count;

	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numKeys)
		return;

	key = keys[idx];
	count = capacity + 1;
	hashedKey = hashFunc(key, capacity);

	while (count) {
		if (hashtable[hashedKey].key == key) {
			deviceResult[idx] = hashtable[hashedKey].value;
			break;
		}

		count--;
		hashedKey++;
		hashedKey %= capacity;
	}
}

__global__ void copy_and_rehash(KeyValue *dst, KeyValue *src,
			unsigned int oldSize, unsigned int newSize) {
	unsigned int idx;
	unsigned int newSlot;
	unsigned int oldVal;

	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= oldSize)
		return;

	newSlot = hashFunc(src[idx].key, newSize);

	while (true) {
		oldVal = atomicCAS(&dst[newSlot].key, KEY_INVALID, src[idx].key);

		if (oldVal == KEY_INVALID) {
			dst[newSlot].value = src[idx].value;
			break;
		}

		++newSlot;
		newSlot %= newSize;
	}
}

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {
	cudaError_t err;

	capacity = size;
	occupancy = 0;

	err = cudaMalloc((void**) &hashtable, size * sizeof(KeyValue));
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMemset((void *) hashtable, KEY_INVALID, size * sizeof(KeyValue));
	DIE(err != cudaSuccess, "cudaMemset");
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {
	cudaError_t err;

	err = cudaFree(hashtable);
	DIE(err != cudaSuccess, "cudaFree");
}

/* RESHAPE HASH
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t err;
	KeyValue *newTable;
	unsigned int numBlocks = (capacity / BLOCK_SIZE) + 1;

	// Malloc device memory and set it as unused
	err = cudaMalloc((void **) &newTable, numBucketsReshape * sizeof(KeyValue));
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMemset((void *) newTable, KEY_INVALID, numBucketsReshape * sizeof(KeyValue));
	DIE(err != cudaSuccess, "cudaMemset");

	// Run kernel
	copy_and_rehash<<< numBlocks, BLOCK_SIZE >>>(newTable, hashtable, capacity, numBucketsReshape);

	// Update hashtable with new data
	err = cudaFree(hashtable);
	DIE(err != cudaSuccess, "cudaFree");

	hashtable = newTable;
	capacity = numBucketsReshape;
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	cudaError_t err;
	int *deviceKeys, *deviceValues;
	size_t numBytes = numKeys * sizeof(int);
	unsigned int numBlocks = (numKeys / BLOCK_SIZE) + 1;

	// Manage load factor
	if (((float) occupancy + (float) numKeys) / (float) capacity >= MAX_LOAD_FACTOR)
		reshape((int) (((float) occupancy + (float) numKeys) / MIN_LOAD_FACTOR));
	
	// Alloc device memory
	err = cudaMalloc((void **) &deviceKeys, numBytes);
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMalloc((void **) &deviceValues, numBytes);
	DIE(err != cudaSuccess, "cudaMalloc");

	// Copy params from host to device
	err = cudaMemcpy(deviceKeys, keys, numBytes, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");
	err = cudaMemcpy(deviceValues, values, numBytes, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");

	// Run kernel and wait for all threads to finish
	insert_entries<<< numBlocks, BLOCK_SIZE >>>(deviceKeys, deviceValues, numKeys, hashtable, capacity);

	// Update structure
	occupancy += numKeys;

	// Free memory
	err = cudaFree(deviceKeys);
	DIE(err != cudaSuccess, "cudaFree");
	err = cudaFree(deviceValues);
	DIE(err != cudaSuccess, "cudaFree");
	
	return true;
}

/* GET BATCH
 */
#ifndef IBM

int* GpuHashTable::getBatch(int* keys, int numKeys) {
	cudaError_t err;
	int *deviceResult;
	int *deviceKeys;
	unsigned int numBlocks = (numKeys / BLOCK_SIZE) + 1;
	size_t numBytes = numKeys * sizeof(int);

	// Malloc device memory (shared memory for the result)
	err = cudaMallocManaged((void **) &deviceResult, numBytes);
	DIE(err != cudaSuccess, "cudaMallocManaged");

	err = cudaMalloc((void **) &deviceKeys, numBytes);
	DIE(err != cudaSuccess, "cudaMalloc");
	
	// Copy params to device memory
	err = cudaMemcpy(deviceKeys, keys, numBytes, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");

	// Run kernel and wait for all the threads to finish
	get_values<<< numBlocks, BLOCK_SIZE >>>(deviceKeys, numKeys, hashtable, capacity, deviceResult);

	err = cudaFree(deviceKeys);
	DIE(err != cudaSuccess, "cudaFree");

	return deviceResult;
}

#else
/* GET BACTCH
 * for IBM-DP.Q
 * NO MALLOCMANAGED
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	cudaError_t err;
	int *deviceResult;
	int *deviceKeys;
	int *result = new int[numKeys];
	unsigned int numBlocks = (numKeys / BLOCK_SIZE) + 1;
	size_t numBytes = numKeys * sizeof(int);

	// Malloc device memory (shared memory for the result)
	err = cudaMalloc((void **) &deviceResult, numBytes);
	DIE(err != cudaSuccess, "cudaMalloc");

	err = cudaMalloc((void **) &deviceKeys, numBytes);
	DIE(err != cudaSuccess, "cudaMalloc");

	// Copy params to device memory
	err = cudaMemcpy(deviceKeys, keys, numBytes, cudaMemcpyHostToDevice);
	DIE(err != cudaSuccess, "cudaMemcpy");

	// Run kernel and wait for all the threads to finish
	get_values<<< numBlocks, BLOCK_SIZE >>>(deviceKeys, numKeys, hashtable, capacity, deviceResult);

	err = cudaMemcpy(result, deviceResult, numBytes, cudaMemcpyDeviceToHost);
	DIE(err != cudaSuccess, "cudaMemcpy");

	err = cudaFree(deviceKeys);
	DIE(err != cudaSuccess, "cudaFree");

	return result;
}
#endif
/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
	// No larger than 1.0f = 100%
	return (float) ((float) occupancy / (float) capacity);
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
