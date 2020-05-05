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
	KeyValue currentPair;

	idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numKeys)
		return;

	key = keys[idx];
	count = capacity;
	hashedKey = hashFunc(key, capacity);

	while (count) {
		currentPair = hashtable[hashedKey];

		if (currentPair.key == key) {
			deviceResult[idx] = currentPair.value;
			break;
		}
		else if (currentPair.key == KEY_INVALID)
			break;

		--count;
		++hashedKey;
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
	capacity = size;
	occupancy = 0;

	cudaMalloc((void**) &hashtable, size * sizeof(KeyValue));
	DIE(hashtable == NULL, "cudaMalloc");

	cudaMemset((void *) hashtable, KEY_INVALID, size * sizeof(KeyValue));
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {
	cudaFree(hashtable);
}

/* RESHAPE HASH
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	// unsigned int numBlocks;
	KeyValue *newTable;

	// numBlocks = MAX(numBucketsReshape / BLOCK_SIZE, 1);

	// Malloc device memory and set it as unused
	cudaMalloc((void **) &newTable, numBucketsReshape * sizeof(KeyValue));
	DIE(newTable == NULL, "cudaMalloc");

	cudaMemset((void *) newTable, KEY_INVALID, numBucketsReshape * sizeof(KeyValue));

	// Run kernel
	copy_and_rehash<<< 1, capacity >>>(newTable, hashtable, capacity, numBucketsReshape);
	cudaDeviceSynchronize();

	// Update hashtable with new data
	cudaFree(hashtable);
	hashtable = newTable;
	capacity = numBucketsReshape;
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	int *deviceKeys, *deviceValues;
	int numBytes = numKeys * sizeof(int);

	// Alloc device memory
	cudaMalloc((void **) &deviceKeys, numBytes);
	DIE(deviceKeys == NULL, "cudaMalloc");

	cudaMalloc((void **) &deviceValues, numBytes);
	DIE(deviceValues == NULL, "cudaMalloc");

	// Check if we have enough space in the table for inserting this chunk
	if ((float) ((float) numKeys + (float) occupancy) / capacity >= MAX_LOAD_FACTOR)
		reshape((int) ((capacity + numKeys) / MIN_LOAD_FACTOR));

	// Copy params from host to device
	cudaMemcpy(deviceKeys, keys, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numBytes, cudaMemcpyHostToDevice);

	// Run kernel and wait for all threads to finish
	insert_entries<<< 1, numKeys >>>(deviceKeys, deviceValues, numKeys, hashtable, capacity);
	cudaDeviceSynchronize();

	// Update structure
	occupancy += numKeys;

	// Free memory
	cudaFree(deviceKeys);
	cudaFree(deviceValues);

	// Manage load factor
	// if (loadFactor() > MAX_LOAD_FACTOR)
	// 	reshape((int) (capacity * 1.8f));
	
	return true;
}

/* GET BATCH
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	// unsigned int numBlocks;
	int *deviceResult = NULL;
	int *deviceKeys = NULL;
	size_t numBytes = numKeys * sizeof(int);

	// numBlocks = MAX(numKeys / BLOCK_SIZE, 1);

	// Malloc device memory (shared memory for the result)
	cudaMallocManaged((void **) &deviceResult, numBytes);
	DIE(deviceResult == NULL, "cudaMalloc");

	cudaMalloc((void **) &deviceKeys, numBytes);
	DIE(deviceKeys == NULL, "cudaMalloc");
	
	// Copy params to device memory
	cudaMemcpy(deviceKeys, keys, numBytes, cudaMemcpyHostToDevice);

	// Run kernel and wait for all the threads to finish
	get_values<<< 1, numKeys >>>(deviceKeys, numKeys, hashtable, capacity, deviceResult);
	cudaDeviceSynchronize();

	cudaFree(deviceKeys);
	return deviceResult;
}

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
