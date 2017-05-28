#include <bitset>
#include "supplement.h"
/*
TODO: 
- Get data off the wire/file and put into a given array/vector
- Extract the date from the packets, send extracted packet data to an array/vector 
- Then send data to the GPU to be hashed out by a given kernel
- 
*/

//-----------------------------------------------------------------------------
//CUDA kernel cuda
//----------------------------------------------------------------------------
__device__ void lock(int* mutex){
  while(atomicCAS(mutex, 0, 1) != 0){

  }
}

__device__ void unlock(int*mutex){
  atomicExch(mutex,0);
}

__global__ void
vectorAdd(const float* A, const float* B, float* C, int numElem){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i< numElem){
    C[i] = A[i] + B[i];
  }
}

__global__ void
DHashData(u_char* pkts, size_t ipitch, u_char* output, size_t opitch, size_t dataLength, int numElem){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  u_char*term = pkts + idx*ipitch;
  u_char*dest = output + idx*opitch;
  if(idx < numElem){
	  sha1Device(term, dataLength, dest);
  }
}

__global__ void
hashSeed(u_int seed, u_int series, u_char* output, size_t opitch, size_t dataLength, int numElem){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  u_int temp = (seed + (idx + (series * numElem)));
  u_char*term = (u_char*) &temp;
  u_char*dest = output + idx*opitch;
  if(idx < numElem){
	  sha1Device(term, dataLength, dest);
  }
}

__global__ void
singleSHA1(u_char* src, int size, u_char* dst){
  sha1Device(src, size, dst);
}

__global__ void
findMatch(u_char* hashed, size_t hashedPitch, int hashedSize, u_char* pool, size_t poolPitch, int poolSize, int hashLength, u_int* result){
}

__global__ void
findMatchV2(u_char* dataPoints, size_t dataPitch, u_int dataSize, u_int* identifiers, u_int identSize, u_int bValue,  u_int maxIDs, u_int throwAway, u_int* result){
  int idx = blockIdx.x * threadIdx.x + blockDim.x;
  if(idx<((dataSize*(maxIDs - throwAway)))){
    int mdPosition = idx / (maxIDs - throwAway);
    u_char* currMD = dataPoints + (mdPosition * MD_LENGTH);
    int identityIndex = idx % (maxIDs -throwAway + 1);
    int tempID = getNthIdentifier((u_int*)currMD, identityIndex, bValue, MD_LENGTH);
    int isS = isSet((u_int*) identifiers, tempID);
//    printf("mdPosition:, %d, identityIndex:, %d, tempID;, %d, isS:, %d\n", mdPosition, identityIndex, tempID, isS);
//    result[tempID] += isSet((u_int*)identifiers, tempID);
    if(isS == 0){
      result[mdPosition] = 1;
    }
  }
}

__global__ void
byteToInt(int* out, u_char* in, int n){
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  for(; idx < n; idx += gridDim.x * blockDim.x){
    out[idx] = in[idx];
  }
}

//The Host does not need this function however for testing purposes, its avaiable to the host
__host__ __device__ int
memcmpDevice(const void* s1, const void* s2, size_t n){
  // ret < 0 if s1 is greater, 0 if equal, ret > 0 if s2 is greater
  int ret = 0;
  u_char* t1 = (u_char*) s1;
  u_char* t2 = (u_char*) s2;
  for(int i = 0; i < n; i++){
   if( (t1 && !t2) || (*t1 > *t2 )) {
      return -1;
    }
    if( (!t1 && t2) || (*t1 < *t2)) {
      return 1;
    }
    t1++;
    t2++;
  }
  return ret;
}

__host__ __device__ void
memcpyDevice(void* dest, const void* src, size_t n){
  u_char* s1 = (u_char*)dest;
  u_char* s2 = (u_char*)src;
  for(int i = 0; i < n; i++){
    *s1 = *s2;
    s1++;
    s2++;
  }
}

extern "C" bool
handleData(u_char* dataPool, int dataLength, int dataPoolSize, 
  u_char* secretPool, int sPoolSize, u_int* ret, struct btimes* times)
{
  bool chk = true;
  u_char* remoteData;
  u_char* remoteHashed;
  u_char* remotePool;
  u_int* remoteResult;
//  u_int* remoteScratch;
  size_t remoteDataPitch = 0;
  size_t remoteHashedPitch = 0;
  size_t remotePoolPitch = 0;
  StopWatchInterface* timer = NULL;
  cudaError_t error;
  sdkCreateTimer(&timer);
  
  //mallocs
  if(times) sdkStartTimer(&timer); 
  checkCudaErrors(cudaMallocPitch((void**)&remoteData, &remoteDataPitch,
		  	  	  	  dataLength*sizeof(u_char), dataPoolSize));
  checkCudaErrors(cudaMallocPitch((void**)&remotePool, &remotePoolPitch,
		  	  	  	  MD_LENGTH*sizeof(u_char), sPoolSize));
  checkCudaErrors(cudaMallocPitch((void**)&remoteHashed, &remoteHashedPitch,
		  	  	  	  MD_LENGTH*sizeof(u_char), dataPoolSize));
  checkCudaErrors(cudaMalloc((void**)&remoteResult, sizeof(u_int)));
  checkCudaErrors(cudaMemset(remoteResult, 0, sizeof(u_int)));
//  checkCudaErrors(cudaMalloc((void**)&remoteScratch, sizeof(u_int)));
//  checkCudaErrors(cudaMemset(remoteScratch, 0, sizeof(u_int)));
  if(times) {
    cudaDeviceSynchronize();
   error = cudaGetLastError();
    sdkStopTimer(&timer);
    times->mallocTime = sdkGetTimerValue(&timer);
    sdkResetTimer(&timer);
    if(error!=cudaSuccess)
    {
      fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
      exit(-1);
    }
  }
  
  //memcopies
  if(times) sdkStartTimer(&timer);
  checkCudaErrors(cudaMemcpy2D(remoteData, remoteDataPitch, dataPool,
		  	  	  	  dataLength*sizeof(u_char), dataLength*sizeof(u_char),
		  	  	  	  dataPoolSize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy2D(remotePool, remotePoolPitch, secretPool,
		  	  	  	  MD_LENGTH*sizeof(u_char), MD_LENGTH*sizeof(u_char),
		  	  	  	  sPoolSize, cudaMemcpyHostToDevice));
  if(times){
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    sdkStopTimer(&timer);
    times->memcpyHTDTime = sdkGetTimerValue(&timer);
    sdkResetTimer(&timer);
    if(error!=cudaSuccess)
    {
      fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
      exit(-1);
    }
  }
  
  //hash the data
  int threadsPerBlock = 512; // for 980ti 1024;
  int blocksPerThread = (dataPoolSize + threadsPerBlock +1) / threadsPerBlock;
  if(times) sdkStartTimer(&timer);
  DHashData<<<threadsPerBlock, blocksPerThread>>>(remoteData, remoteDataPitch, remoteHashed,
		  remoteHashedPitch, dataLength*sizeof(u_char), dataPoolSize);
  if(times){
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    times->hashTime = sdkGetTimerValue(&timer);
    sdkResetTimer(&timer);
  }
  
  //compare the hashes against the pool
  int threadsPerBlock2 = 512;
  int blocksPerGrid2 = ((sPoolSize * dataPoolSize) + threadsPerBlock +1)/threadsPerBlock;
  if(times) sdkStartTimer(&timer);
  findMatch<<<blocksPerGrid2, threadsPerBlock2>>>(remoteHashed,
		  remoteHashedPitch, dataPoolSize, remotePool, remotePoolPitch,
		  sPoolSize, MD_LENGTH, remoteResult);
  if(times) {
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    sdkStopTimer(&timer);
    times->findTime = sdkGetTimerValue(&timer);
    sdkResetTimer(&timer);
    if(error!=cudaSuccess)
    {
      fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
      exit(-1);
    } 
  }

  //return the int (bit array?)
  if(times) sdkStartTimer(&timer);
  checkCudaErrors(cudaMemcpy(ret, remoteResult, sizeof(u_int),
		  cudaMemcpyDeviceToHost));
  if(times){
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    sdkStopTimer(&timer);
    times->memcpyDTHTime = sdkGetTimerValue(&timer);
    times->totalTime = times->mallocTime + times->hashTime + times->findTime +
    				   times->memcpyDTHTime + times->memcpyHTDTime;
    if(error!=cudaSuccess)
    {
      fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
      exit(-1);
    }
  }
  //free and clean up memory
  cudaFree(remoteData);
  cudaFree(remoteHashed);
  cudaFree(remotePool);
  cudaFree(remoteResult);
//  cudaFree(remoteScratch);
  sdkDeleteTimer(&timer);
  return chk;
}

__host__ __device__ int
getNthIdentifier(u_int* buff, int nth, int bValue, int length){
	u_int *temp = buff;
	u_int bytesPerInt = 4;
	u_int bitsPerByte = 8;
	u_int maxBit = length*bytesPerInt*bitsPerByte; //maximum number of bits that could be in the buff
	u_int maxNumberIdenties = maxBit / bValue;
	int nthStartBit = nth * bValue;
	u_int nthEndBit = nthStartBit + bValue - 1;
	u_int startIntIndex = nthStartBit / 31; //this will be the unsigned
								   //int which will contain our first bit.
	u_int startIntRangeBeg = startIntIndex * 32;
	u_int accum = 0;
	u_int exp = bValue -1;
	u_int startIntBitIndex = nthStartBit - startIntRangeBeg;
	if(nth > maxNumberIdenties || nth < 0 ||
			nthEndBit > maxBit ||
			nthStartBit < 0 ||
			bValue <= 0){ return -1;}
	//get the int closest to the first bit we need
	//if need get the rest of the bits
	for(int i = exp; i >= 0; i--){
		accum += ((temp[startIntIndex] >> (31 - startIntBitIndex)) & 1) << i;
		//check if still in range
		//iterate to next bit
		if((startIntBitIndex + 1) > 31){
		  startIntIndex++;
		  startIntBitIndex = 0;
		} else {
		  startIntBitIndex++;
		}
	}
	return accum;
}

extern "C" bool
handleDataV2(u_char* dataPool, u_int dataLength, u_int dataPoolSize,
  u_int* secretPool, u_int sPoolSize, u_int bValue, u_int maxIDs, u_int throwAway, u_int* ret, struct btimes* times)
{
  bool chk = true;
  u_char* remoteData;
  u_char* remoteHashed;
  u_int* remotePool;
  u_int* remoteResult;
  u_int* scratchResult;
  size_t remoteDataPitch = 0;
  size_t remoteHashedPitch = 0;
  StopWatchInterface* timer = NULL;
  cudaError_t error;
  sdkCreateTimer(&timer);
  
  //mallocs
  if(times) sdkStartTimer(&timer); 
  checkCudaErrors(cudaMallocPitch((void**)&remoteData, &remoteDataPitch,
		  	  	  	  dataLength*sizeof(u_char), dataPoolSize));
  checkCudaErrors(cudaMallocPitch((void**)&remoteHashed, &remoteHashedPitch,
		  	  	  	  MD_LENGTH*sizeof(u_char), dataPoolSize));
  checkCudaErrors(cudaMalloc((void**)&remotePool, sizeof(u_int)*sPoolSize));
  checkCudaErrors(cudaMalloc((void**)&remoteResult,sizeof(u_int)*dataPoolSize));
  checkCudaErrors(cudaMemset(remoteResult, 0, sizeof(u_int)*dataPoolSize));
  scratchResult = new u_int[dataPoolSize];
  if(times) {
    cudaDeviceSynchronize();
   error = cudaGetLastError();
    sdkStopTimer(&timer);
    times->mallocTime = sdkGetTimerValue(&timer);
    sdkResetTimer(&timer);
    if(error!=cudaSuccess)
    {
      fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
      exit(-1);
    }
  }
  
  //memcopies
  if(times) sdkStartTimer(&timer);
  checkCudaErrors(cudaMemcpy2D(remoteData, remoteDataPitch, dataPool,
		  	  	  	  dataLength*sizeof(u_char), dataLength*sizeof(u_char),
		  	  	  	  dataPoolSize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(remotePool, secretPool, sizeof(int)*sPoolSize, cudaMemcpyHostToDevice));
  if(times){
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    sdkStopTimer(&timer);
    times->memcpyHTDTime = sdkGetTimerValue(&timer);
    sdkResetTimer(&timer);
    if(error!=cudaSuccess)
    {
      fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
      exit(-1);
    }
  }
  
  //hash the data
  int threadsPerBlock = 512;
  int blocksPerThread = (dataPoolSize + threadsPerBlock -1) / threadsPerBlock;
//  printf("bValue: %d tValue: %d maxIDs: %d, sPoolSize: %d\n", bValue, throwAway, maxIDs, sPoolSize);
/*  for(int p = 0; p < sPoolSize; p++){
    printf("[%d]", secretPool[p]);
  }
  printf("\n");*/
  if(times) sdkStartTimer(&timer);
  DHashData<<<blocksPerThread, threadsPerBlock>>>(remoteData, remoteDataPitch, remoteHashed,
		  remoteHashedPitch, dataLength*sizeof(u_char), dataPoolSize);
  if(times){
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    times->hashTime = sdkGetTimerValue(&timer);
    sdkResetTimer(&timer);
  }
  
  //compare the hashes against the pool
  int threadsPerBlock2 = 1024;
  int blocksPerGrid2 = ((dataPoolSize * (maxIDs - throwAway)) + threadsPerBlock -1)/threadsPerBlock;
  if(times) sdkStartTimer(&timer);
//  printf("check value before launch\n");
/*  for(int c = 0; c < sPoolSize; c++){                                        
/*
  printf("check value before launch\n");
  for(int c = 0; c < sPoolSize; c++){                                        
    bitset<32>t1(secretPool[c]);                                              
    printf("|%s_%d|", t1.to_string().c_str(),secretPool[c]);                  
  }   
  printf("\n"); 
  */ 
  findMatchV2<<<blocksPerGrid2, threadsPerBlock2>>>(remoteHashed,
		  remoteHashedPitch, dataPoolSize, remotePool, sPoolSize, bValue, maxIDs, throwAway, remoteResult);
  if(times) {
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    sdkStopTimer(&timer);
    times->findTime = sdkGetTimerValue(&timer);
    sdkResetTimer(&timer);
    if(error!=cudaSuccess)
    {
      fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
      exit(-1);
    } 
  }

  //return the int (bit array?)
  if(times) sdkStartTimer(&timer);
  checkCudaErrors(cudaMemcpy(scratchResult, remoteResult,dataPoolSize,
		  cudaMemcpyDeviceToHost));
  int succMessCount = 0;
  for(int x=0; x <= dataPoolSize; x++){
    succMessCount += (scratchResult[x])?0:1;
  }
  if(times){
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    sdkStopTimer(&timer);
    times->memcpyDTHTime = sdkGetTimerValue(&timer);
    times->totalTime = times->mallocTime + times->hashTime + times->findTime +
    				   times->memcpyDTHTime + times->memcpyHTDTime;
    if(error!=cudaSuccess)
    {
      fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
      exit(-1);
    }
  }
  //free and clean up memory
  delete[] scratchResult;
  cudaFree(remoteData);
  cudaFree(remoteHashed);
  cudaFree(remotePool);
  cudaFree(remoteResult);
  sdkDeleteTimer(&timer);
  return chk;
}

extern "C" bool
handleDataV3(u_int seed, u_int numThreads, u_int* secretPool, u_int sPoolSize, u_int bValue, u_int maxIDs, u_int throwAway, u_int* ret, struct btimes* times)
{
  bool chk = true;
  bool broken = false;
  int series = 0;
  u_char* remoteHashed;
  u_int* remotePool;
  u_int* remoteResult;
  u_int* scratchResult;
  size_t remoteHashedPitch = 0;
  StopWatchInterface* timer = NULL;
  cudaError_t error;
  sdkCreateTimer(&timer);

  while(broken == false){
    //mallocs
    if(times) sdkStartTimer(&timer); 
    checkCudaErrors(cudaMallocPitch((void**)&remoteHashed, &remoteHashedPitch, MD_LENGTH*sizeof(u_char), numThreads));
    checkCudaErrors(cudaMalloc((void**)&remotePool, sizeof(u_int)*sPoolSize));
    checkCudaErrors(cudaMalloc((void**)&remoteResult,sizeof(u_int)*numThreads));
    checkCudaErrors(cudaMemset(remoteResult, 0, sizeof(u_int)*numThreads));
    scratchResult = new u_int[numThreads];
    if(times) {
      cudaDeviceSynchronize();
      error = cudaGetLastError();
      sdkStopTimer(&timer);
      times->mallocTime = sdkGetTimerValue(&timer);
      sdkResetTimer(&timer);
      if(error!=cudaSuccess)
      {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
      }
    }

    //copy
    if(times) sdkStartTimer(&timer);
    checkCudaErrors(cudaMemcpy(remotePool, secretPool, sizeof(int)*sPoolSize, cudaMemcpyHostToDevice));
    if(times){
      cudaDeviceSynchronize();
      error = cudaGetLastError();
      sdkStopTimer(&timer);
      times->memcpyHTDTime = sdkGetTimerValue(&timer);
      sdkResetTimer(&timer);
      if(error!=cudaSuccess)
      {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
      }
    }

    int threadsPerBlock = 512;
    int blocksPerThread = (numThreads + threadsPerBlock -1) / threadsPerBlock;
    if(times) sdkStartTimer(&timer);
    hashSeed<<<blocksPerThread, threadsPerBlock>>>(seed, series, remoteHashed,
        remoteHashedPitch, numThreads*sizeof(u_char), numThreads);
    if(times){
      cudaDeviceSynchronize();
      sdkStopTimer(&timer);
      times->hashTime = sdkGetTimerValue(&timer);
      sdkResetTimer(&timer);
    }

    //compare the hashes against the pool
    int threadsPerBlock2 = 1024;
    int blocksPerGrid2 = ((numThreads * (maxIDs - throwAway)) + threadsPerBlock -1)/threadsPerBlock;
    if(times) sdkStartTimer(&timer);
    findMatchV2<<<blocksPerGrid2, threadsPerBlock2>>>(remoteHashed,
        remoteHashedPitch, numThreads, remotePool, sPoolSize, bValue, maxIDs, throwAway, remoteResult);
    if(times) {
      cudaDeviceSynchronize();
      error = cudaGetLastError();
      sdkStopTimer(&timer);
      times->findTime = sdkGetTimerValue(&timer);
      sdkResetTimer(&timer);
      if(error!=cudaSuccess)
      {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
      }
    }
    //return the int (bit array?)
    if(times) sdkStartTimer(&timer);
    checkCudaErrors(cudaMemcpy(scratchResult, remoteResult,numThreads*sizeof(u_int), cudaMemcpyDeviceToHost));
    printf("what\n");
    int succMessCount = 0;
    for(int x=0; x <= numThreads; x++){
      printf("%d %d\n", x,scratchResult[x]);
      succMessCount += (scratchResult[x])?0:1;
    }
    printf("-\n");
    if(times){
      cudaDeviceSynchronize();
      error = cudaGetLastError();
      sdkStopTimer(&timer);
      times->memcpyDTHTime = sdkGetTimerValue(&timer);
      times->totalTime = times->mallocTime + times->hashTime + times->findTime +
        times->memcpyDTHTime + times->memcpyHTDTime;
      if(error!=cudaSuccess)
      {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);
      }
    }
    if(succMessCount > 0){ 
      fprintf(stderr, "broken\n");
      broken = true;
    }
    series++;
    delete[] scratchResult;
    cudaFree(remoteHashed);
    cudaFree(remotePool);
    cudaFree(remoteResult);
    sdkDeleteTimer(&timer);


  }
  //free
  //return
  return chk;
}


extern "C" bool
sha1Kernel2D(u_char* src, int pWidth, int pHeight, u_char* dst,
	struct btimes* times)
{
  bool ret = true;
  size_t pitchSrc = 0;
  size_t pitchDst = 0;
  size_t widthSrc = pWidth * sizeof(u_char);
  size_t widthDst = MD_LENGTH * sizeof(u_char);
  size_t height = pHeight;
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  u_char* remoteSrc = NULL;
  u_char* remoteDst = NULL;  


/*
  sdkStartTimer(&timer);
  checkCudaErrors(cudaMalloc((void**) &remoteSrc,  widthSrc * height));
  checkCudaErrors(cudaMalloc((void**) &remoteDst, widthDst * height));
  checkCudaErrors(cudaMemcpy(remoteSrc, src, widthSrc * height, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(remoteDst, src, widthDst * height, cudaMemcpyHostToDevice));
  sdkStopTimer(&timer);
 // printf("malloc and copy [%f ms]\n", sdkGetTimerValue(&timer));
  sdkResetTimer(&timer);
*/

  //2d array implementation
  if(times) sdkStartTimer(&timer);
  checkCudaErrors(cudaMallocPitch((void**)&remoteSrc,&pitchSrc, widthSrc, height));
  checkCudaErrors(cudaMallocPitch((void**)&remoteDst,&pitchDst, widthDst, height));
  if(times){
    sdkStopTimer(&timer);
    times->mallocTime = sdkGetTimerValue(&timer);
    sdkResetTimer(&timer);
  }

  //memcpy
  if(times) sdkStartTimer(&timer);
  checkCudaErrors(cudaMemcpy2D(remoteSrc, pitchSrc, src, widthSrc * sizeof(u_char), 
                                widthSrc * sizeof(u_char), height, cudaMemcpyHostToDevice));
  if(times){
    sdkStopTimer(&timer);
    times->memcpyHTDTime = sdkGetTimerValue(&timer);
    sdkResetTimer(&timer);
  }
  
  //kernel launch, hash data

  int threadsPerBlock = 128;
  int blocksPerGrid = (pHeight + threadsPerBlock -1) / threadsPerBlock;
  if(times) sdkStartTimer(&timer);
  //  flattenHashData<<<height, 1>>>(remoteSrc, widthSrc, remoteDst, widthDst, height);
  DHashData<<<blocksPerGrid, threadsPerBlock>>>(remoteSrc, pitchSrc, remoteDst, pitchDst, widthSrc, height);
  if(times){
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    times->hashTime = sdkGetTimerValue(&timer);
    sdkResetTimer(&timer);
  } 
 
  //copy data back from host 
  if(times) sdkStartTimer(&timer);
  checkCudaErrors(cudaMemcpy2D(dst, widthDst * sizeof(u_char), remoteDst, pitchDst,
                                widthDst * sizeof(u_char), height, cudaMemcpyDeviceToHost));
  if(times){
    cudaDeviceSynchronize();
    sdkStopTimer(&timer); 
    times->memcpyDTHTime = sdkGetTimerValue(&timer);
    sdkResetTimer(&timer);
  }

  checkCudaErrors(cudaFree(remoteSrc));
  checkCudaErrors(cudaFree(remoteDst));
  sdkDeleteTimer(&timer);
  return ret;
}

extern "C" bool
sha1KernelPerSecond(u_char* src, int pWidth, int pHeight, u_char* dst,
	struct btimes* times, u_int * hashes)
{
  bool ret = true;
  size_t pitchSrc = 0;
  size_t pitchDst = 0;
  size_t widthSrc = pWidth * sizeof(u_char);
  size_t widthDst = MD_LENGTH * sizeof(u_char);
  size_t height = pHeight;
  int repeats = 5000;
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  u_char* remoteSrc = NULL;
  u_char* remoteDst = NULL;  

  //2d array implementation
  checkCudaErrors(cudaMallocPitch((void**)&remoteSrc,&pitchSrc, widthSrc, height));
  checkCudaErrors(cudaMallocPitch((void**)&remoteDst,&pitchDst, widthDst, height));

  //memcpy
  checkCudaErrors(cudaMemcpy2D(remoteSrc, pitchSrc, src, widthSrc * sizeof(u_char), 
                                widthSrc * sizeof(u_char), height, cudaMemcpyHostToDevice));
  
  //kernel launch, hash data
  int threadsPerBlock = 128;
  int blocksPerGrid = (pHeight + threadsPerBlock -1) / threadsPerBlock;
  if(times) sdkStartTimer(&timer);
  //  flattenHashData<<<height, 1>>>(remoteSrc, widthSrc, remoteDst, widthDst, height);
  for(int x = 0; x < repeats; x++){
    DHashData<<<blocksPerGrid, threadsPerBlock>>>(remoteSrc, pitchSrc, remoteDst, pitchDst, widthSrc, height);
    *hashes += height;
  }
  if(times){
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    times->totalTime = (sdkGetTimerValue(&timer));
    sdkResetTimer(&timer);
  } 
 
  //copy data back from host 
  checkCudaErrors(cudaMemcpy2D(dst, widthDst * sizeof(u_char), remoteDst, pitchDst,
                                widthDst * sizeof(u_char), height, cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(remoteSrc));
  checkCudaErrors(cudaFree(remoteDst));
  sdkDeleteTimer(&timer);
  return ret;
}

extern "C" bool
hashFindPerSecond(u_int* hashes, struct btimes* bench){
  int target = 45056;
  int maxRange = 1000;
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  u_char* data = new u_char[target * sizeof(int)];
  memset(data, 0, (target* sizeof(int)));
  u_char* check = new u_char[target * MD_LENGTH];
  memset(check, 0, (target*MD_LENGTH*sizeof(u_char)));
  u_char* out = new u_char[target * MD_LENGTH];
  memset(out, 0, (target*MD_LENGTH*sizeof(u_char)));
  int* temp = (int*)data;
  u_int bValue = 4;
  u_int maxIds = (1 << bValue) - 1;
  u_int identitiesPerMD = (INTS_PER_MD * BITS_PER_INT) / bValue;
  u_int throwAway = 0;
  for(int i = 0; i < target; i++){
    int ranNum = 1 + (rand() % maxRange);
    temp[i] = ranNum;
  }

  //hash all the random values into a checking array;
  for(int i = 0; i < target; i++){
    SHA1(data + (i*sizeof(int)), sizeof(int), check + (i*MD_LENGTH));
  }

  int checkSize = (maxIds / 32);
  u_int* pool = new u_int[checkSize];
  memset(pool, 0, checkSize*sizeof(u_int)); 
  u_char* currMD = NULL;
  for(int i = 0; i< (target); i++){
    currMD = check + (i + MD_LENGTH);
    for(int j = 0; j < identitiesPerMD; j++){
      int x = getNthIdentifier((u_int*) currMD, j, bValue, MD_LENGTH);
      setBit(pool, x);
    }
  }
  //hash the random values on the GPU
  handleDataPerSecond(data, (u_int) sizeof(int), (u_int) target, (u_int*) pool, 
      (u_int) target, bValue, maxIds, throwAway, (u_int*) hashes, bench); 
  delete[] data;
  delete[] out;
  delete[] check;
  sdkDeleteTimer(&timer);;
  return true;
}

extern "C" bool
hashesPerSecond(u_int* hashes,struct btimes* bench){
  int target = 45056/2;
  int maxRange = 1000;
  size_t height = target;
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  u_char* data = new u_char[target * sizeof(int)];
  memset(data, 0, (target* sizeof(int)));
  u_char* check = new u_char[target * MD_LENGTH];
  memset(check, 0, (target*MD_LENGTH*sizeof(u_char)));
  u_char* out = new u_char[target * MD_LENGTH];
  memset(out, 0, (target*MD_LENGTH*sizeof(u_char)));
  int* temp = (int*)data;

  for(int i = 0; i < target; i++){
    int ranNum = 1 + (rand() % maxRange);
    temp[i] = ranNum;
  }

  //hash all the random values into a checking array;
  for(int i = 0; i < target; i++){
    SHA1(data + (i*sizeof(int)), sizeof(int), check + (i*MD_LENGTH));
  }

  //hash the random values on the GPU
  sha1KernelPerSecond(data, sizeof(int), height, out, bench, hashes);
  
  delete[] data;
  delete[] out;
  delete[] check;
  sdkDeleteTimer(&timer);;
  return true;
}

extern "C" bool
handleDataPerSecond(u_char* dataPool, u_int dataLength, u_int dataPoolSize,
  u_int* secretPool, u_int sPoolSize, u_int bValue, u_int maxIDs, u_int throwAway, u_int* ret,struct btimes* times)
{
  bool chk = true;
  u_char* remoteData;
  u_char* remoteHashed;
  u_int* remotePool;
  u_int* remoteResult;
  u_int* scratchResult;
  size_t remoteDataPitch = 0;
  size_t remoteHashedPitch = 0;
  StopWatchInterface* timer = NULL;
  cudaError_t error;
  int repeats = 5000;
  sdkCreateTimer(&timer);
  
  //mallocs
  checkCudaErrors(cudaMallocPitch((void**)&remoteData, &remoteDataPitch,
		  	  	  	  dataLength*sizeof(u_char), dataPoolSize));
  checkCudaErrors(cudaMallocPitch((void**)&remoteHashed, &remoteHashedPitch,
		  	  	  	  MD_LENGTH*sizeof(u_char), dataPoolSize));
  checkCudaErrors(cudaMalloc((void**)&remotePool, sizeof(u_int)*sPoolSize));
  checkCudaErrors(cudaMalloc((void**)&remoteResult, sizeof(u_int)*(maxIDs+1)));
  checkCudaErrors(cudaMemset(remoteResult, 0, sizeof(u_int)*(maxIDs+1)));
  scratchResult = new u_int[maxIDs+1];

  //memcopies
  checkCudaErrors(cudaMemcpy2D(remoteData, remoteDataPitch, dataPool,
		  	  	  	  dataLength*sizeof(u_char), dataLength*sizeof(u_char),
		  	  	  	  dataPoolSize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(remotePool, secretPool, sizeof(int)*sPoolSize, cudaMemcpyHostToDevice));
  
  if(times) sdkStartTimer(&timer);
  for(int x  = 0; x < repeats; x++){
    int threadsPerBlock = 128; //1024; for the 980ti
    int blocksPerThread = (dataPoolSize + threadsPerBlock -1) / threadsPerBlock;
    DHashData<<<blocksPerThread, threadsPerBlock>>>(remoteData, remoteDataPitch, remoteHashed,
		  remoteHashedPitch, dataLength*sizeof(u_char), dataPoolSize); 
    cudaDeviceSynchronize();
    //compare the hashes against the pool
    int threadsPerBlock2 =  128;
    int blocksPerGrid2 = ((dataPoolSize * (maxIDs - throwAway)) + threadsPerBlock -1)/threadsPerBlock;
    findMatchV2<<<blocksPerGrid2, threadsPerBlock2>>>(remoteHashed,
		  remoteHashedPitch, dataPoolSize, remotePool, sPoolSize, bValue, maxIDs, throwAway, remoteResult);
    cudaDeviceSynchronize();
    *ret += dataPoolSize;
  }
  if(times) {
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    sdkStopTimer(&timer);
    times->totalTime = sdkGetTimerValue(&timer);
    sdkResetTimer(&timer);
    if(error!=cudaSuccess)
    {
      fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
      exit(-1);
    } 
  }

  //return the int (bit array?)
  checkCudaErrors(cudaMemcpy(scratchResult, remoteResult,sizeof(int)*(maxIDs+1),
		  cudaMemcpyDeviceToHost));
  
  //free and clean up memory
  delete[] scratchResult;
  cudaFree(remoteData);
  cudaFree(remoteHashed);
  cudaFree(remotePool);
  cudaFree(remoteResult);
  sdkDeleteTimer(&timer);
  return chk;
}

extern "C" bool
sha1Kernel(u_char* src, int len, u_char* dest){
  bool ret = false;
  size_t size = len* sizeof(u_char);
  u_char* remoteSrc = NULL;
  u_char* remoteDst = NULL; 
  
  checkCudaErrors(cudaMalloc((void**)&remoteSrc, size));
  checkCudaErrors(cudaMalloc((void**)&remoteDst, MD_LENGTH*sizeof(u_char)));
  checkCudaErrors(cudaMemcpy(remoteSrc, src, size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(remoteDst, dest, MD_LENGTH*sizeof(u_char), cudaMemcpyHostToDevice));

  singleSHA1<<<1,1>>>(remoteSrc, size, remoteDst);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(dest, remoteDst, 20*sizeof(u_char),
		  cudaMemcpyDeviceToHost));
  

  //check against openssl's sha1

  u_char tempHash[20];
  SHA1(src, len, tempHash);
  
  checkCudaErrors(cudaFree(remoteSrc));
  checkCudaErrors(cudaFree(remoteDst));
  ret = (0 == memcmp(tempHash, dest, 20));
  return ret;
  
}



__host__ __device__ void
setBit(u_int* x, u_int val){
  u_int index = val / 32;
  u_int shift = 31 - (val % 32);
  x[index] |= 1 << shift;
}

__host__ __device__ void
clearBit(u_int* x, u_int val){
  u_int index = val / 32;
  u_int shift = 31 - (val % 32);
  x[index] &= ~(1 << shift);
}

__host__ __device__ void
toggleBit(u_int* x, u_int val){
  u_int index = val / 32;
  u_int shift = 31 - (val % 32);
  x[index] ^= 1 << shift;
}

__host__ __device__ int
isSet(u_int* x, u_int val){
  u_int index = val / 32;
  u_int shift = 31 - (val % 32);
  return (x[index] >> shift) & 1;
}

extern "C" bool 
vectorAdditionExample(const int argc, const char **argv, float *arrA, float *arrB, float *output, int len){
  bool ret = true;
  size_t size = len * sizeof(float);
  float* remoteA = NULL;
  float* remoteB = NULL;
  float* remoteOutput = NULL;

  //Create memory on the remote device
#ifdef DEBUG  
  printf("Allocate memory on the remote device\n");
#endif
  checkCudaErrors(cudaMalloc((void**)&remoteA, size));
  checkCudaErrors(cudaMalloc((void**)&remoteB, size));
  checkCudaErrors(cudaMalloc((void**)&remoteOutput, size));

  //Copy data to remote device
#ifdef DEBUG
  printf("Copy memory from host to remote device\n"); 
#endif
  checkCudaErrors(cudaMemcpy(remoteA, arrA, size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(remoteB, arrB, size, cudaMemcpyHostToDevice));

  //Launch Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
#ifdef DEBUG  
  printf("Launch Kernel\n");
#endif
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(remoteA, remoteB, remoteOutput, size);
  checkCudaErrors(cudaGetLastError());

  //Copy output from remote device
#ifdef DEBUG 
  printf("Copy output data from the CUDA device to the host memory\n"); 
#endif
  checkCudaErrors(cudaMemcpy(output, remoteOutput, size, cudaMemcpyDeviceToHost));

  for(int i = 0; i < len; i++){
    if(fabs(arrA[i] + arrB[i] - output[i]) > 1e-5){
      fprintf(stderr, "Result verification failed at a element %d\n", i);
      ret = false;
    }
  }

  //Free the remote data 
  checkCudaErrors(cudaFree(remoteA));
  checkCudaErrors(cudaFree(remoteB));
  checkCudaErrors(cudaFree(remoteOutput));

  return ret;	
}

/*
 * SHA-1 CPU implementation
 */
__device__ const unsigned char sha1_padding[64] =
{
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};


/*
 * Prepare SHA-1 for execution.
 */
__host__ __device__ void 
sha1Init(unsigned long* total, unsigned long* state, unsigned char* buff)
{
        total[0] = 0;
        total[1] = 0;
        state[0] = 0x67452301;
        state[1] = 0xEFCDAB89;
        state[2] = 0x98BADCFE;
        state[3] = 0x10325476;
        state[4] = 0xC3D2E1F0;
}


/*
 * Process one block of data.
 */
__host__ __device__ void 

sha1ProcessBlock(unsigned long* total, unsigned long* state, unsigned char* buff, unsigned char data[64])
{
        unsigned long temp, W[16]={0,}, A, B, C, D, E;
  
        GET_UINT32_BE( W[ 0], data,  0 );
        GET_UINT32_BE( W[ 1], data,  4 );
        GET_UINT32_BE( W[ 2], data,  8 );
        GET_UINT32_BE( W[ 3], data, 12 );
        GET_UINT32_BE( W[ 4], data, 16 );
        GET_UINT32_BE( W[ 5], data, 20 );
        GET_UINT32_BE( W[ 6], data, 24 );
        GET_UINT32_BE( W[ 7], data, 28 );
        GET_UINT32_BE( W[ 8], data, 32 );
        GET_UINT32_BE( W[ 9], data, 36 );
        GET_UINT32_BE( W[10], data, 40 );
        GET_UINT32_BE( W[11], data, 44 );
        GET_UINT32_BE( W[12], data, 48 );
        GET_UINT32_BE( W[13], data, 52 );
        GET_UINT32_BE( W[14], data, 56 );
        GET_UINT32_BE( W[15], data, 60 );
  
#define S(x,n) ((x << n) | ((x & 0xFFFFFFFF) >> (32 - n)))

#define R(t)                                            \
(                                                       \
    temp = W[(t -  3) & 0x0F] ^ W[(t - 8) & 0x0F] ^     \
           W[(t - 14) & 0x0F] ^ W[ t      & 0x0F],      \
    ( W[t & 0x0F] = S(temp,1) )                         \
)

#define P(a,b,c,d,e,x)                                  \
{                                                       \
    e += S(a,5) + F(b,c,d) + K + x; b = S(b,30);        \
}

        A = state[0];
        B = state[1];
        C = state[2];
        D = state[3];
        E = state[4];
  
#define F(x,y,z) (z ^ (x & (y ^ z)))
#define K 0x5A827999
  
        P( A, B, C, D, E, W[0]  );
        P( E, A, B, C, D, W[1]  );
        P( D, E, A, B, C, W[2]  );
        P( C, D, E, A, B, W[3]  );
        P( B, C, D, E, A, W[4]  );
        P( A, B, C, D, E, W[5]  );
        P( E, A, B, C, D, W[6]  );
        P( D, E, A, B, C, W[7]  );
        P( C, D, E, A, B, W[8]  );
        P( B, C, D, E, A, W[9]  );
        P( A, B, C, D, E, W[10] );
        P( E, A, B, C, D, W[11] );
        P( D, E, A, B, C, W[12] );
        P( C, D, E, A, B, W[13] );
        P( B, C, D, E, A, W[14] );
        P( A, B, C, D, E, W[15] );
        P( E, A, B, C, D, R(16) );
        P( D, E, A, B, C, R(17) );
        P( C, D, E, A, B, R(18) );
        P( B, C, D, E, A, R(19) );

#undef K
#undef F

#define F(x,y,z) (x ^ y ^ z)
#define K 0x6ED9EBA1
  
        P( A, B, C, D, E, R(20) );
        P( E, A, B, C, D, R(21) );
        P( D, E, A, B, C, R(22) );
        P( C, D, E, A, B, R(23) );
        P( B, C, D, E, A, R(24) );
        P( A, B, C, D, E, R(25) );
        P( E, A, B, C, D, R(26) );
        P( D, E, A, B, C, R(27) );
        P( C, D, E, A, B, R(28) );
        P( B, C, D, E, A, R(29) );
        P( A, B, C, D, E, R(30) );
        P( E, A, B, C, D, R(31) );
        P( D, E, A, B, C, R(32) );
        P( C, D, E, A, B, R(33) );
        P( B, C, D, E, A, R(34) );
        P( A, B, C, D, E, R(35) );
        P( E, A, B, C, D, R(36) );
        P( D, E, A, B, C, R(37) );
        P( C, D, E, A, B, R(38) );
        P( B, C, D, E, A, R(39) );

#undef K
#undef F

#define F(x,y,z) ((x & y) | (z & (x | y)))
#define K 0x8F1BBCDC

        P( A, B, C, D, E, R(40) );
        P( E, A, B, C, D, R(41) );
        P( D, E, A, B, C, R(42) );
        P( C, D, E, A, B, R(43) );
        P( B, C, D, E, A, R(44) );
        P( A, B, C, D, E, R(45) );
        P( E, A, B, C, D, R(46) );
        P( D, E, A, B, C, R(47) );
        P( C, D, E, A, B, R(48) );
        P( B, C, D, E, A, R(49) );
        P( A, B, C, D, E, R(50) );
        P( E, A, B, C, D, R(51) );
        P( D, E, A, B, C, R(52) );
        P( C, D, E, A, B, R(53) );
        P( B, C, D, E, A, R(54) );
        P( A, B, C, D, E, R(55) );
        P( E, A, B, C, D, R(56) );
        P( D, E, A, B, C, R(57) );
        P( C, D, E, A, B, R(58) );
        P( B, C, D, E, A, R(59) );

#undef K
#undef F

#define F(x,y,z) (x ^ y ^ z)
#define K 0xCA62C1D6
  
        P( A, B, C, D, E, R(60) );
        P( E, A, B, C, D, R(61) );
        P( D, E, A, B, C, R(62) );
        P( C, D, E, A, B, R(63) );
        P( B, C, D, E, A, R(64) );
        P( A, B, C, D, E, R(65) );
        P( E, A, B, C, D, R(66) );
        P( D, E, A, B, C, R(67) );
        P( C, D, E, A, B, R(68) );
        P( B, C, D, E, A, R(69) );
        P( A, B, C, D, E, R(70) );
        P( E, A, B, C, D, R(71) );
        P( D, E, A, B, C, R(72) );
        P( C, D, E, A, B, R(73) );
        P( B, C, D, E, A, R(74) );
        P( A, B, C, D, E, R(75) );
        P( E, A, B, C, D, R(76) );
        P( D, E, A, B, C, R(77) );
        P( C, D, E, A, B, R(78) );
        P( B, C, D, E, A, R(79) );

#undef K
#undef F

        state[0] += A;
        state[1] += B;
        state[2] += C;
        state[3] += D;
        state[4] += E;
}


/*
 * Splits input message into blocks and processes them one by one. Also
 * checks how many 0 need to be padded and processes the last, padded, block.
 */
__host__ __device__ void 
sha1Update(unsigned long* total, unsigned long* state, unsigned char* buff, unsigned char *input, int ilen)
{
        int fill;
        unsigned long left;
  
        if ( ilen <= 0 )
                return;
  
        left = total[0] & 0x3F;
        fill = 64 - left;
  
        total[0] += ilen;
        total[0] &= 0xFFFFFFFF;

        if (total[0] < (unsigned long) ilen)
                total[1]++;
  
        if ( left && ilen >= fill ) {
                memcpy((void *) (buff + left), (void *) input, fill);
                sha1ProcessBlock(total, state, buff, buff);
                input += fill;
                ilen  -= fill;
                left = 0;
        }
  
        while ( ilen >= 64 ) {
                sha1ProcessBlock(total, state, buff, input);
                input += 64;
                ilen  -= 64;
        }
  
        if ( ilen > 0 ) {
                memcpy( (void *) (buff + left), (void *) input, ilen );
        }
}


/*
 * Process padded block and return hash to user.
 */
__host__ __device__ void 
sha1Finish(unsigned long* total,unsigned long* state, unsigned char* buff, unsigned char *output)
{

        unsigned long last, padn;
        unsigned long high, low;
        unsigned char msglen[8];


        high = (total[0] >> 29) | (total[1] <<  3);
        low  = (total[0] <<  3);

        PUT_UINT32_BE(high, msglen, 0);
        PUT_UINT32_BE(low,  msglen, 4);

        last = total[0] & 0x3F;
        padn = (last < 56 ) ? ( 56 - last ) : ( 120 - last);

        sha1Update(total, state, buff, (unsigned char *) sha1_padding, padn);
        sha1Update(total, state, buff, msglen, 8);

        PUT_UINT32_BE(state[0], output,  0);
        PUT_UINT32_BE(state[1], output,  4);
        PUT_UINT32_BE(state[2], output,  8);
        PUT_UINT32_BE(state[3], output, 12);
        PUT_UINT32_BE(state[4], output, 16);
}

/*
 * Execute SHA-1
 */

__host__ __device__ void
sha1Device(unsigned char *input, int ilen, unsigned char *output) {
        unsigned long total[2];
        unsigned long state[5];
        unsigned char buff[64];

        sha1Init( total, state, buff );
        sha1Update( total, state, buff, input, ilen );
        sha1Finish( total, state, buff, output );

}


string getCurrGpuTemp(){
  system("./updateCurrentGpuTemp");
  string ret;
  ifstream infile("./.currGpuTemp");
  if(infile.good()){
    getline(infile, ret);
  }
  return ret;
}
