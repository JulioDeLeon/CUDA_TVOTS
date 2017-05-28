#include "supplement.h"
#include <bitset>
using namespace std;
//http://cpptest.sourceforge.net/tutorial.html
void setBtimesTotal(struct btimes *x){
  x->totalTime = x->mallocTime + x->memcpyDTHTime + x->memcpyHTDTime 
    + x->hashTime + x->findTime;
}


void initBtimes(struct btimes *x){
  x->mallocTime = 0.0;
  x->memcpyDTHTime = 0.0;
  x->memcpyHTDTime = 0.0;
  x->hashTime = 0.0;
  x->findTime = 0.0;
  x->totalTime = 0.0;
}



class CudaKernelTestSuite : public Test::Suite {
public:
  CudaKernelTestSuite(){
//    TEST_ADD(CudaKernelTestSuite::first_test)
//    TEST_ADD(CudaKernelTestSuite::PacketClassDataTest)
//    TEST_ADD(CudaKernelTestSuite::GpuMemcmpTest)
//    TEST_ADD(CudaKernelTestSuite::GpuMemcpyTest)
//    TEST_ADD(CudaKernelTestSuite::SHA1GpuTest)
//    TEST_ADD(CudaKernelTestSuite::SHA1GpuTest2D)
//    TEST_ADD(CudaKernelTestSuite::SHA1GpuTest2DComplex)
//    TEST_ADD(CudaKernelTestSuite::SHA1CPUTest)
//    TEST_ADD(CudaKernelTestSuite::GPU2dHashFindBench)
//    TEST_ADD(CudaKernelTestSuite::GPU2dHashFind)
//    TEST_ADD(CudaKernelTestSuite::MockAttack)
//    TEST_ADD(CudaKernelTestSuite::GetIdentifierTest)
//    TEST_ADD(CudaKernelTestSuite::HashPerSecondTest)
//    TEST_ADD(CudaKernelTestSuite::HashFindPerSecondTest);
//   TEST_ADD(CudaKernelTestSuite::BitTest);
//    TEST_ADD(CudaKernelTestSuite::MockAttack1024);
//    TEST_ADD(CudaKernelTestSuite::MockAttack2048);
//    TEST_ADD(CudaKernelTestSuite::MockAttack4096);
    TEST_ADD(CudaKernelTestSuite::MockAttack8192);
  }
private:  
  void first_test();
  void PacketClassDataTest();
  void SHA1GpuTest();
  void SHA1GpuTest2D();
  void SHA1GpuTest2DComplex();
  void GpuMemcmpTest();
  void GpuMemcpyTest();
  void SHA1CPUTest();
  void GPU2dHashFind();
  void GPU2dHashFindBench();
  void GetIdentifierTest();
  void HashPerSecondTest();
  void HashFindPerSecondTest();
  void BitTest();
  void MockAttack1024();
  void MockAttack2048();
  void MockAttack4096();
  void MockAttack8192();
};

void CudaKernelTestSuite::first_test(){
  clock_t start, end;
  double elapsed;
  start = time(NULL);
  // test timing of functions here
  end = time(NULL);
  elapsed = difftime(start, end);
  TEST_ASSERT(1 == 1); //CHnage this assert to check if functions have past some time benchmark
}

void CudaKernelTestSuite::SHA1CPUTest(){
  u_char hash1[20] = {};
  u_char hash2[20] = {};
  u_char t1[] = "abc";
  u_char e1[] = "\xa9\x99\x3e\x36\x47\x06\x81\x6a\xba\x3e\x25\x71\x78\x50\xc2\x6c\x9c\xd0\xd8\x9d";
  u_char t2[] =  "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  u_char e2[] = "\x84\x98\x3e\x44\x1c\x3b\xd2\x6e\xba\xae\x4a\xa1\xf9\x51\x29\xe5\xe5\x46\x70\xf1";
  
  
  SHA1(t1, strlen((const char*) t1), hash1);
  sha1Device(t1, strlen((const char*)t1), hash2); 
  TEST_ASSERT(0 == memcmp(e1, hash1, 20)); 
  TEST_ASSERT(0 == memcmp(e1, hash2, 20)); 
  
  SHA1(t2, strlen((const char*) t2), hash1);
  sha1Device(t2, strlen((const char*) t2), hash2); 
  TEST_ASSERT(0 == memcmp(e2, hash1, 20)); 
  TEST_ASSERT(0 == memcmp(e2, hash2, 20)); 
  
}

void CudaKernelTestSuite::SHA1GpuTest2D(){
  int target = 10;
  u_char t1[] = "abc";
  u_char e1[] = "\xa9\x99\x3e\x36\x47\x06\x81\x6a\xba\x3e\x25\x71\x78\x50\xc2\x6c\x9c\xd0\xd8\x9d";

  u_char data1[target * (strlen((const char*) t1))];
  memset(data1, 0, (target* strlen((const char*) t1)));
  u_char out1[target * MD_LENGTH];
  memset(out1, 0, (target*MD_LENGTH*sizeof(u_char)));

  //fill input
  u_char* temp = data1;
  int tlen = strlen((const char*) t1);
  for(int i = 0; i < target; i++){
    memcpy(temp + (i*tlen),  t1, tlen);
  }
  u_char hash[20];
  SHA1(data1, tlen, hash);

  TEST_ASSERT(0 == memcmp(hash, e1, MD_LENGTH));
  //do the hashing of data set 1
   u_char hash2[20];
  sha1Kernel(data1, tlen, hash2);

  TEST_ASSERT(0 == memcmp(hash2, e1, MD_LENGTH));


  sha1Kernel2D(data1, tlen, target, out1, NULL);
  u_char* tempRep = new u_char[20];
  for(int i = 0; i < target; i++){
	  memcpy(tempRep, out1 + (i * MD_LENGTH), MD_LENGTH);
	  TEST_ASSERT(0 == memcmp(tempRep, e1, MD_LENGTH));
  }


  delete[] tempRep;
  //compare the results of hash 1 with expected
  //do the hashing of data set 2


  TEST_ASSERT(0==0);
}

void CudaKernelTestSuite::SHA1GpuTest2DComplex(){
  int target = 10000;
  int maxRange = 1000;
  struct btimes avgGpuTimes;
  struct btimes cpuTimes;
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  u_char* data = new u_char[target * sizeof(int)];
  memset(data, 0, (target* sizeof(int)));
  u_char* check = new u_char[target * MD_LENGTH];
  memset(check, 0, (target*MD_LENGTH*sizeof(u_char)));
  u_char* out = new u_char[target * MD_LENGTH];
  memset(out, 0, (target*MD_LENGTH*sizeof(u_char)));
  initBtimes(&cpuTimes);
  initBtimes(&avgGpuTimes);  
  int* temp = (int*)data;
  u_char* temp2 = check;
  u_char* temp3 = out;

  for(int i = 0; i < target; i++){
	  //memcpy(temp + (i*tlen),  t1, tlen);
	  //This time fill data array with random values;
	  int ranNum = 1 + (rand() % maxRange);
	  temp[i] = ranNum;
  }

  sdkStartTimer(&timer);
  //hash all the random values into a checking array;
  for(int i = 0; i < target; i++){
	  SHA1(data + (i*sizeof(int)), sizeof(int), check + (i*MD_LENGTH));
  }
  sdkStopTimer(&timer);
  cpuTimes.hashTime = sdkGetTimerValue(&timer);
  sdkResetTimer(&timer);

  //printing array
  for(int i = 0; i < target; i++){
	  u_char tempMsg[MD_LENGTH];
	  memcpy(tempMsg, temp2 +(i*MD_LENGTH), MD_LENGTH);
	  //printf("CPU[%d] => [%s]\n", temp[i], GetHexRepresentation(tempMsg, MD_LENGTH).c_str() );
  }
  //hash the random values on the GPU
  sha1Kernel2D(data, sizeof(int), target, out, &avgGpuTimes);

  for(int i = 0; i < target; i++){
	  u_char tempMsg[MD_LENGTH];
	  memcpy(tempMsg, temp2 +(i*MD_LENGTH), MD_LENGTH);
          //printf("GPU[%d] => [%s]\n", temp[i], GetHexRepresentation(tempMsg, MD_LENGTH).c_str() );
  }

  //checking array
  for(int i = 0; i < target; i++){
	  u_char correctMsg[MD_LENGTH]; //we assume any sha1 hash produced by openssl is the target we want
	  memcpy(correctMsg, temp2 +(i*MD_LENGTH), MD_LENGTH);

	  u_char gpuMsg[MD_LENGTH];
	  memcpy(gpuMsg, temp2 +(i*MD_LENGTH), MD_LENGTH);
	  TEST_ASSERT(0 == memcmp(correctMsg, gpuMsg, MD_LENGTH));
  }
  cpuTimes.totalTime = cpuTimes.hashTime + cpuTimes.findTime; 
  avgGpuTimes.totalTime = avgGpuTimes.hashTime + avgGpuTimes.findTime + avgGpuTimes.mallocTime + avgGpuTimes.memcpyDTHTime + avgGpuTimes.memcpyHTDTime;

  delete[] data;
  delete[] out;
  delete[] check;
  sdkDeleteTimer(&timer);
  cudaDeviceReset();
}

void CudaKernelTestSuite::PacketClassDataTest(){
  //char filename[] = "tvots_gs_packets.pcapng";
  char filename[] = "8192-40-filtered.pcapng";
  const u_char* packet;
  char* tmp;
  int errBuffSize = 1024;
  struct pcap_pkthdr header;
  char* errBuff = new char[errBuffSize];
  pcap_t* pcapH = pcap_open_offline(filename, errBuff);
  int check_target = 50000;
  int pktcnt = 0;

  TEST_ASSERT(pcapH != NULL);
  do{
    packet = pcap_next(pcapH, &header);
    if(header.len != TARGET_PACKET_LENGTH) continue;
    if(!packet) continue;
    tmp = (char*) packet;
    TVOTS_Packet* captured = new TVOTS_Packet(packet);
    //check fields of the new packet to the original.
    
    //data
    TEST_ASSERT(0 ==  memcmp(packet + DATA_POSITION, captured->getEventData(), DATA_LENGTH));
    //timestamp
    TEST_ASSERT(0 == memcmp(packet + TIMESTAMP_POSITION, captured->getTimeStamp(), TIMESTAMP_LENGTH));
    //epoch
    TEST_ASSERT(0 == memcmp(packet + EPOCH_ID_POSITION, captured->getEpochID(), EPOCH_ID_LENGTH));
    //secret
    TEST_ASSERT(0 == memcmp(packet + TIMESTAMP_POSITION, captured->getTimeStamp(), TIMESTAMP_LENGTH));
    //chain hash id
    TEST_ASSERT(0 == memcmp(packet + HASHCHAIN_ID_POSITION, captured->getHashChainID(), HASHCHAIN_ID_LENGTH));
    //ksecrets
    u_char** k_secrets = captured->getKSecrets();
    for(int i = 0; i < NUMBER_K_SECRETS; i++){
      TEST_ASSERT(0 == memcmp(k_secrets[i], packet + K_SECRET_POSITION + (i * K_SECRET_LENGTH), K_SECRET_LENGTH));  
    }

    pktcnt++;
  }while(packet && pktcnt < check_target);
  
  delete errBuff;
}

void CudaKernelTestSuite::SHA1GpuTest(){
  u_char t1[] = "abc";
  u_char e1[] = "\xa9\x99\x3e\x36\x47\x06\x81\x6a\xba\x3e\x25\x71\x78\x50\xc2\x6c\x9c\xd0\xd8\x9d";
  u_char t2[] = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"; 
  u_char e2[] = "\x84\x98\x3e\x44\x1c\x3b\xd2\x6e\xba\xae\x4a\xa1\xf9\x51\x29\xe5\xe5\x46\x70\xf1";
  u_char hash1[20];
  u_char hash2[20];

  //openssl is used for reference to test correctness of the inhouse sha1 implemenations;
  SHA1(t1, strlen((const char*) t1), hash1);
  TEST_ASSERT(0 == memcmp(hash1, e1, 20));
  //gpu hash1  
  TEST_ASSERT(sha1Kernel(t1, strlen((const char*) t1), hash2));
  TEST_ASSERT(0 == memcmp(hash1, hash2, 20));

  SHA1(t2, strlen((const char*) t2), hash1);
  TEST_ASSERT(0 == memcmp(hash1, e2, 20));
  //gpu hash 2
  TEST_ASSERT(sha1Kernel(t2, strlen((const char*) t2), hash2));
  TEST_ASSERT(0 == memcmp(hash1, hash2, 20));
  cudaDeviceReset();
  
}

void CudaKernelTestSuite::GpuMemcmpTest(){
//remote comment from insight
  char filename[] = "tvots_gs_packets.pcapng";
  const u_char* packet;
  char* tmp;
  int errBuffSize = 1024;
  struct pcap_pkthdr header;
  char* errBuff = new char[errBuffSize];
  pcap_t* pcapH = pcap_open_offline(filename, errBuff);
  int check_target = 1;
  int pktcnt = 0;

  TEST_ASSERT(pcapH != NULL);
  do{
    packet = pcap_next(pcapH, &header);
    if(header.len != TARGET_PACKET_LENGTH) continue;
    if(!packet) continue;
    tmp = (char*) packet;
    TVOTS_Packet* captured = new TVOTS_Packet(packet);
    //check fields of the new packet to the original.
    
    //data
    TEST_ASSERT(0 ==  memcmp(packet + DATA_POSITION, captured->getEventData(), DATA_LENGTH));
    //timestamp
    TEST_ASSERT(0 == memcmp(packet + TIMESTAMP_POSITION, captured->getTimeStamp(), TIMESTAMP_LENGTH));
    //epoch
    TEST_ASSERT(0 == memcmp(packet + EPOCH_ID_POSITION, captured->getEpochID(), EPOCH_ID_LENGTH));
   
    //hashchain 
    TEST_ASSERT(0 == memcmp(packet + HASHCHAIN_ID_POSITION, captured->getHashChainID(), HASHCHAIN_ID_LENGTH));
    
    //secret
    TEST_ASSERT(0 == memcmp(packet + TIMESTAMP_POSITION, captured->getTimeStamp(), TIMESTAMP_LENGTH));
    
    //ksecrets
    u_char** k_secrets = captured->getKSecrets();
    for(int i = 0; i < NUMBER_K_SECRETS; i++){
      TEST_ASSERT(0 == memcmp(k_secrets[i], packet + K_SECRET_POSITION + (i * K_SECRET_LENGTH), K_SECRET_LENGTH));  
    }

    //improper data size test
    TEST_ASSERT( 0 != memcmp(captured->getEventData(), "a", DATA_LENGTH));
    TEST_ASSERT( 0 != memcmp(captured->getEventData(), "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", DATA_LENGTH));

    pktcnt++;
  }while(packet && pktcnt < check_target);
  
  delete errBuff;
}

void CudaKernelTestSuite::GpuMemcpyTest(){
  char t1[] = "012345678901234567890";
  char* d1 = new char[strlen(t1)];

  memcpy(d1, t1, strlen(t1));
  TEST_ASSERT(0 == memcmp(t1, d1, strlen(t1)));
  TEST_ASSERT(0 != memcmp(d1, "123123235234", strlen(d1)));
}


void CudaKernelTestSuite::GPU2dHashFind(){
  u_int startTarget = 1;
  u_int target = startTarget;
  int secretMultiple = 14; //This number is based off a ratio of number of secrets to data points
  int sampleSize = 1;
  u_int bValue = 10;
  u_int maxIdentity = (1 << bValue) - 1;
  u_int identitiesPerMD = (INTS_PER_MD * BITS_PER_INT) / bValue;
  u_int throwAway = 0;
  int maxRange = 1000000;
  int targetIncrements = 1;
  int bValueIterations = 5;
  int checkSize = 0;
  u_int testCount = 0;
  u_char *data, *gpuH, *cpuH, *pool;
  for(int w = 0; w < bValueIterations; w++){
    for(int x = 0; x < targetIncrements; x++){ 
      for(int y = 0; y < sampleSize; y++){
        data = new u_char[target * sizeof(int)];
        memset(data, 0, (target* sizeof(int)));
        gpuH = new u_char[target * MD_LENGTH];
        memset(gpuH, 0, (target*MD_LENGTH*sizeof(u_char)));
        cpuH = new u_char[target * MD_LENGTH];
        memset(cpuH, 0, (target*MD_LENGTH*sizeof(u_char)));
        pool = new u_char[secretMultiple*target * MD_LENGTH];
        memset(pool, 0, (secretMultiple*target*MD_LENGTH*sizeof(u_char)));
        int ranDataIndex =  (rand() % target); //This will be a random index which our hard example t1 will be inserted in our data pool
        int ranCheckIndex =  (rand() % (target*secretMultiple)); //This will be the random index which out hard check e1 will be inserted in our secrets
        u_char t1[] = "abc";
        u_int tlen = strlen((const char*) t1);
        u_char e1[] = "\xa9\x99\x3e\x36\x47\x06\x81\x6a\xba\x3e\x25\x71\x78\x50\xc2\x6c\x9c\xd0\xd8\x9d";

        //init data  and poolData
        u_int* temp = (u_int*)data;
        for(int i = 0; i < target/sizeof(u_int); i++){
          u_int ranNum = 1 + (rand() % maxRange);
          temp[i] = ranNum;
        }

        //insert hard example
        memcpy(data + (ranDataIndex * tlen), t1, tlen);

        temp = (u_int*)pool;
        for(int i = 0; i < (target*secretMultiple*INTS_PER_MD); i++){
          //memcpy(temp + (i*tlen),  t1, tlen);
          //This time fill data array with random values;
          u_int ranNum = 1 + (rand() % maxRange);
          temp[i] = ranNum;
        }

        //insert hard check
        memcpy(pool + (ranCheckIndex*MD_LENGTH), e1, MD_LENGTH);
        //now fill the indentifiers into the check;
        checkSize = (maxIdentity / (sizeof(int) * BITS_PER_BYTE))+1;
        u_int* check = new u_int[checkSize];
        memset(check, 0, checkSize*sizeof(u_int));
        u_char* currMD = NULL;
        for(int i = 0; i < (target * secretMultiple); i++){
          currMD = pool + (i*MD_LENGTH);
          for(int j = 0; j < identitiesPerMD - throwAway; j++){
            int x = getNthIdentifier((u_int*) currMD, j, bValue, MD_LENGTH / sizeof(u_int));
            setBit(check, x);
          }
        }
        //hash the data on cpu
        for(int i = 0; i < target; i++){
          SHA1(data + (i*tlen), tlen, cpuH + (i*MD_LENGTH));
        }
        
        //find hash in pool cpu SIDE
        bool found = false;
        u_char* tempMD = NULL;
        u_int* cpuResult = new u_int[checkSize];
        memset(cpuResult, 0, checkSize * sizeof(u_int));
        for(int i = 0; i < target; i++){
          tempMD = cpuH + (i*MD_LENGTH);
          for(int j = 0; j < identitiesPerMD - throwAway; j++){
            int x = getNthIdentifier((u_int*) tempMD, j, bValue, MD_LENGTH / sizeof(u_int));
            if(isSet(check, x)){
              setBit(cpuResult, x);
              found = true;
            }
          }
        }
        
        //hash data on GPU and find match
        u_int gpuResult = 0;
        //tlen will be the size of a given data point in bytes
        handleDataV2(data, tlen, target, check, checkSize, bValue, identitiesPerMD, throwAway, &gpuResult,NULL);
        delete[] data;
        delete[] gpuH;
        delete[] cpuH;
        delete[] pool;
        delete[] check;
        checkSize = 0;
        maxIdentity = 0;
      }
      target = (target == 1)?256:target<<1;
    }
    target = startTarget;
    bValue++;
    maxIdentity = (1 << bValue) - 1;
    identitiesPerMD = (INTS_PER_MD * BITS_PER_INT) / bValue;
  }
  TEST_ASSERT( 0 == 0);

}

bool BenchTemplate(FILE* log, u_int bVal, u_int tVal, int targetIterations, int samplep, struct tm* timep){
	  u_int target = 1;
	  int secretMultiple = 14; //This number is based off a ratio of number of secrets to data points
	  int sampleSize = samplep;
	  u_int bValue = bVal;
	  u_int maxIdentity = (1 << bValue) - 1;
	  u_int identitiesPerMD = (INTS_PER_MD * BITS_PER_INT) / bValue;
	  u_int throwAway = tVal;
	  int maxRange = 1000000;
	  int iterations = targetIterations;
	  int checkSize = (maxIdentity / (sizeof(int) * BITS_PER_BYTE))+1;
	  struct btimes avgGpuTimes;
	  struct btimes cpuTimes;
	  struct btimes avgCpuTimes;
	  struct btimes gpuTimes;
	  StopWatchInterface *timer = NULL;
	  sdkCreateTimer(&timer);
	  srand(time(NULL));
	  u_int testCount = 0;
	  u_char *data, *gpuH, *cpuH, *pool;
	  struct tm* timeInfo = timep;
	time_t rawTime;
	time(&rawTime);
	  FILE* logfile = log;

	  for(int x = 0; x < iterations; x++){
		initBtimes(&gpuTimes);
	    initBtimes(&cpuTimes);
	    initBtimes(&avgGpuTimes);
	    initBtimes(&avgCpuTimes);
	    for(int y = 0; y < sampleSize; y++){
	      data = new u_char[target * sizeof(int)];
	      memset(data, 0, (target* sizeof(int)));
	      gpuH = new u_char[target * MD_LENGTH];
	      memset(gpuH, 0, (target*MD_LENGTH*sizeof(u_char)));
	      cpuH = new u_char[target * MD_LENGTH];
	      memset(cpuH, 0, (target*MD_LENGTH*sizeof(u_char)));
	      pool = new u_char[secretMultiple*target * MD_LENGTH];
	      memset(pool, 0, (secretMultiple*target*MD_LENGTH*sizeof(u_char)));

	      int ranDataIndex =  (rand() % target); //This will be a random index which our hard example t1 will be inserted in our data pool
	      int ranCheckIndex =  (rand() % (target*secretMultiple)); //This will be the random index which out hard check e1 will be inserted in our secrets
	      u_char t1[] = "abc";
	      u_int tlen = strlen((const char*) t1);
	      u_char e1[] = "\xa9\x99\x3e\x36\x47\x06\x81\x6a\xba\x3e\x25\x71\x78\x50\xc2\x6c\x9c\xd0\xd8\x9d";

	      //init data  and poolData
	      u_int* temp = (u_int*)data;
	      for(int i = 0; i < target/sizeof(u_int); i++){
	        u_int ranNum = 1 + (rand() % maxRange);
	        temp[i] = ranNum;
	      }

	      //insert hard example
	      memcpy(data + (ranDataIndex * tlen), t1, tlen);

	      temp = (u_int*)pool;
	      for(int i = 0; i < (target*secretMultiple*INTS_PER_MD); i++){
	        //memcpy(temp + (i*tlen),  t1, tlen);
	        //This time fill data array with random values;
	        u_int ranNum = 1 + (rand() % maxRange);
	        temp[i] = ranNum;
	      }

	      //insert hard check
	      memcpy(pool + (ranCheckIndex*MD_LENGTH), e1, MD_LENGTH);

	      //now fill the indentifiers into the check;

	      u_int* check = new u_int[checkSize];
	      memset(check, 0, checkSize*sizeof(u_int));
	      u_char* currMD = NULL;
	      for(int i = 0; i < (target * secretMultiple); i++){
	    	currMD = pool + (i*MD_LENGTH);
	        for(int j = 0; j < identitiesPerMD - throwAway; j++){
	          int x = getNthIdentifier((u_int*) currMD, j, bValue, MD_LENGTH / sizeof(u_int));
	          setBit(check, x);
	        }
	      }
	      //hash the data on cpu
	      sdkStartTimer(&timer);
	      for(int i = 0; i < target; i++){
	        SHA1(data + (i*tlen), tlen, cpuH + (i*MD_LENGTH));
	      }
	      sdkStopTimer(&timer);
	      cpuTimes.hashTime = sdkGetTimerValue(&timer);
	      sdkResetTimer(&timer);

	      //find hash in pool cpu SIDE
	      bool found = false;
	      sdkStartTimer(&timer);
	      u_char* tempMD = NULL;
	      u_int* cpuResult = new u_int[checkSize];
	      memset(cpuResult, 0, checkSize * sizeof(u_int));
	      for(int i = 0; i < target; i++){
	    	  tempMD = cpuH + (i*MD_LENGTH);
	    	  for(int j = 0; j < identitiesPerMD - throwAway; j++){
	    		  int x = getNthIdentifier((u_int*) tempMD, j, bValue, MD_LENGTH / sizeof(u_int));
	    		  if(isSet(check, x)){
	    			  setBit(cpuResult, x);
	    			  found = true;
	    		  }
	    	  }
	      }
	      sdkStopTimer(&timer);
	      cpuTimes.findTime = sdkGetTimerValue(&timer);
	      sdkResetTimer(&timer);
	      setBtimesTotal(&cpuTimes);

	      //hash data on GPU and find match
	      u_int gpuResult = 0;
	      //tlen will be the size of a given data point in bytes
	      handleDataV2(data, tlen, target, check, checkSize, bValue, identitiesPerMD, throwAway, &gpuResult, &gpuTimes);

	      //compare the CPU results and gpuResults
	      /*for(int i = 0; i < checkSize; i++){
	    	  bitset<32>x(cpuResult[i]);
	    	  cout << x;
	      }
	      cout << endl;

	      for(int i = 0; i < checkSize; i++){
	    	  bitset<32>x(gpuResult[i]);
	    	  cout << x;
	      }
	      cout << endl;*/
	        avgCpuTimes.totalTime += cpuTimes.totalTime;
	        avgCpuTimes.findTime += cpuTimes.findTime;
	        avgCpuTimes.hashTime += cpuTimes.hashTime;
	        avgCpuTimes.totalTime += cpuTimes.totalTime;
	        avgCpuTimes.mallocTime += cpuTimes.mallocTime;
	        avgCpuTimes.memcpyDTHTime += cpuTimes.memcpyDTHTime;
	        avgCpuTimes.memcpyHTDTime += cpuTimes.memcpyHTDTime;
	        fprintf(logfile,"%d,%s,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%s",
	            ++testCount, "cpu",bValue,throwAway,target,target*secretMultiple,
	            cpuTimes.mallocTime,cpuTimes.memcpyHTDTime,cpuTimes.hashTime,
	            cpuTimes.findTime,cpuTimes.memcpyDTHTime,cpuTimes.totalTime,
	            asctime(timeInfo));

	        time(&rawTime);
	        timeInfo = localtime(&rawTime);
	        avgGpuTimes.totalTime += gpuTimes.totalTime;
	        avgGpuTimes.findTime += gpuTimes.findTime;
	        avgGpuTimes.hashTime += gpuTimes.hashTime;
	        avgGpuTimes.mallocTime += gpuTimes.mallocTime;
	        avgGpuTimes.memcpyDTHTime += gpuTimes.memcpyDTHTime;
	        avgGpuTimes.memcpyHTDTime += gpuTimes.memcpyHTDTime;
	        fprintf(logfile,"%d,%s,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%s",
	            ++testCount, "gpu",bValue,throwAway,target,target*secretMultiple,
	            gpuTimes.mallocTime,gpuTimes.memcpyHTDTime,gpuTimes.hashTime,
	            gpuTimes.findTime,gpuTimes.memcpyDTHTime,gpuTimes.totalTime,
	            asctime(timeInfo));
	      delete[] gpuH;
	      delete[] cpuH;
	      delete[] cpuResult;
	      delete[] pool;
	      delete[] check;

	    }
	      avgCpuTimes.hashTime = avgCpuTimes.hashTime / sampleSize;
	      avgCpuTimes.findTime = avgCpuTimes.findTime / sampleSize;
	      avgCpuTimes.totalTime = avgCpuTimes.hashTime + avgCpuTimes.findTime;
	      avgGpuTimes.mallocTime = avgGpuTimes.mallocTime / sampleSize;
	      avgGpuTimes.findTime = avgGpuTimes.findTime / sampleSize;
	      avgGpuTimes.hashTime = avgGpuTimes.hashTime / sampleSize;
	      avgGpuTimes.memcpyDTHTime = avgGpuTimes.memcpyDTHTime / sampleSize;
	      avgGpuTimes.memcpyHTDTime = avgGpuTimes.memcpyHTDTime / sampleSize;
	      avgGpuTimes.totalTime = avgGpuTimes.mallocTime + avgGpuTimes.memcpyDTHTime +
	        avgGpuTimes.memcpyHTDTime + avgGpuTimes.hashTime +
	        avgGpuTimes.findTime;
	      time(&rawTime);
	      timeInfo = localtime(&rawTime);
	      fprintf(logfile,"%dsample,%s,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%s",
	          sampleSize,"avg cpu",bValue,throwAway,target,target*secretMultiple,
	          avgCpuTimes.mallocTime,avgCpuTimes.memcpyHTDTime,avgCpuTimes.hashTime,
	          avgCpuTimes.findTime,avgCpuTimes.memcpyDTHTime,avgCpuTimes.totalTime,
	          asctime(timeInfo));
	      time(&rawTime);
	      timeInfo = localtime(&rawTime);
	      fprintf(logfile,"%dsample,%s,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%s",
	          sampleSize,"avg gpu",bValue,throwAway,target,target*secretMultiple,
	          avgGpuTimes.mallocTime,avgGpuTimes.memcpyHTDTime,avgGpuTimes.hashTime,
	          avgGpuTimes.findTime,avgGpuTimes.memcpyDTHTime,avgGpuTimes.totalTime,
	          asctime(timeInfo));
	    	target = (target == 1)?256:target<<1;

	  }
	  sdkDeleteTimer(&timer);
	  return true;
}

void CudaKernelTestSuite::GPU2dHashFindBench(){
	 struct tm* timeInfo;
	 time_t rawTime;
	 time(&rawTime);
	 timeInfo = localtime(&rawTime);
	 char fileName[50] = {};
	  sprintf(fileName, "./logs/HashFindBenchLog-%d_%d_%d-%d_%d_%d.csv",
	    timeInfo->tm_year + 1900, timeInfo->tm_mon + 1, timeInfo->tm_mday, timeInfo->tm_hour,
	    timeInfo->tm_min, timeInfo->tm_sec);
	  FILE* logfile = fopen(fileName, "w");
	  //fprintf(logfile, "%s %s\n", __PRETTY_FUNCTION__, asctime(timeInfo));
	  fprintf(logfile, "\"test number\",device,bValue,tValue,\"data points\",\
	secrets,\"malloc time (ms)\",\"memcpyHTD time (ms)\",\"hash time (ms)\",\
	\"find time (ms)\",\"memcpyDTH time (ms)\",\"total time (ms)\",timestamp\n");
	u_int startBVal = 10;
	u_int startTVal = 0;
	u_int bValIterations = 5;
	u_int tValIterations = 1;
	int targetIterations = 10;
	int sampleSize = 10;
	cudaFree(0);
	for(int x = 0; x < bValIterations; x++){
		for(int y = 0; y < tValIterations; y++){
			TEST_ASSERT(BenchTemplate(logfile,startBVal+x,startTVal+y,targetIterations,sampleSize,timeInfo));
		}
	}
	fclose(logfile);
}

void CudaKernelTestSuite::GetIdentifierTest(){
	const u_int dummyInt = 2863311530;
	//this int is for the pattern "10101010..1010" repeating to 32 bits
	u_char* buff = new u_char[MD_LENGTH];
	//conversion is to be consistent with real data
	int bValue, nth, ret, expected;
	u_int* temp = (u_int*) buff;
	for(int i = 0; i < MD_LENGTH / sizeof(u_int); i++){
		temp[i] = dummyInt;
	}
/*
 * | 10101010101010101010101010101010 | x5
 * | 10101010101010101010101010101010 | 10101010101010101010101010101010 |...
 */
	//this test will go against different bValues and different identifiers
	//in of range
	bValue = 7;
	nth = 15;
	ret = getNthIdentifier(temp, nth, bValue, MD_LENGTH/sizeof(u_int));
	expected = 42;
	TEST_ASSERT(ret == expected);

	bValue = 10;
	nth = 3;
	ret = getNthIdentifier(temp, nth, bValue, MD_LENGTH/sizeof(u_int));
	expected = 682;
	TEST_ASSERT(ret == expected);

	bValue = 12;
	nth = 5;
	ret = getNthIdentifier(temp, nth, bValue, MD_LENGTH/sizeof(u_int));
	expected = 2730;
	TEST_ASSERT(ret == expected);

	//out of range
	bValue = 50;
	nth = 7;
	ret = getNthIdentifier(temp, nth, bValue, MD_LENGTH/sizeof(u_int));
	TEST_ASSERT(ret < 0); //this test has an integer overflow

	bValue = 1000;
	nth = 0;
	ret = getNthIdentifier(temp, nth, bValue, MD_LENGTH/sizeof(u_int));
	expected = -1;
	TEST_ASSERT(ret == expected);

	bValue = 12;
	nth = 13;
	ret = getNthIdentifier(temp, nth, bValue, MD_LENGTH/sizeof(u_int));
	expected = -1;
	TEST_ASSERT(ret == expected);

	delete[] buff;
	TEST_ASSERT(0==0);
}

void CudaKernelTestSuite::BitTest(){
  printf("%s\n",__PRETTY_FUNCTION__);
  int bValue = 3;
  int checkSize = 3;
  srand(time(NULL));
  u_int pool[checkSize];
  for(int c = 0; c < checkSize; c++){
    pool[c] = rand() % 10000000;
    bitset<32>t(pool[c]);
    printf("|%s|", t.to_string().c_str());
  }
  printf("\n");
  int x = 44;
  printf("%d is %s\n", x, (isSet((u_int*)pool, x))?"set":"not set");
  setBit(pool, 44);
  for(int c = 0; c < checkSize; c++){
    bitset<32>t(pool[c]);
    printf("|%s|", t.to_string().c_str());
  }
  printf("\n");
 
}

void CudaKernelTestSuite::HashPerSecondTest(){
  time_t rawTime;
  struct tm* timeInfo;
  time(&rawTime);
  timeInfo = localtime(&rawTime);
  FILE* logfile;
  const char* fileName = "./logs/HashPerSecondTest.csv";
  char output[100];
  u_int hashes = 0;
  struct btimes bench;
  initBtimes(&bench);
  hashesPerSecond(&hashes, &bench);
  float per = hashes/bench.totalTime;
  sprintf(output, "%d/%d/%d %d:%d:%d,%d,%f,%f,%s", 
    timeInfo->tm_year + 1900, timeInfo->tm_mon + 1, timeInfo->tm_mday, timeInfo->tm_hour, 
    timeInfo->tm_min, timeInfo->tm_sec, hashes, bench.totalTime, per, getCurrGpuTemp().c_str());
  mkdir("./logs", S_IRWXU | S_IRWXG | S_IRWXO);
  logfile = fopen(fileName, "a");
  fprintf(logfile, "%s\n", output);
 
  fclose(logfile); 
}

void CudaKernelTestSuite::HashFindPerSecondTest(){
  time_t rawTime;
  struct tm* timeInfo;
  time(&rawTime);
  timeInfo = localtime(&rawTime);
  FILE* logfile;
  const char* fileName = "./logs/HashFindPerSecondTest.csv";
  char output[100];
  u_int hashes = 0;
  struct btimes bench;
  initBtimes(&bench);
  hashFindPerSecond(&hashes, &bench);
  float per = hashes/bench.totalTime;
  sprintf(output, "%d/%d/%d %d:%d:%d,%d,%f,%f,%s", 
    timeInfo->tm_year + 1900, timeInfo->tm_mon + 1, timeInfo->tm_mday, timeInfo->tm_hour, 
    timeInfo->tm_min, timeInfo->tm_sec, hashes, bench.totalTime, per,getCurrGpuTemp().c_str());
  logfile = fopen(fileName, "a");
  fprintf(logfile, "%s\n", output);
 
  fclose(logfile); 
}

bool mockAttackTemplate(char* dataDir, int bvlR, u_int tVal, int pckLim){
  time_t rawTime;
  struct tm* timeInfo;
  time(&rawTime);
  timeInfo = localtime(&rawTime);
  FILE* logfile = NULL;
  pcap_t* pcapH = NULL;
  struct pcap_pkthdr header;
  int secretsSeen = 0;
  int pktcnt = 0;
  int packetLimit = pckLim;
  double bvl = log2(bvlR);
  u_int bValue = static_cast<u_int>(bvl + 0.5);
  u_int tValue = tVal;
  u_int maxID = (1 << bValue) - 1;
  u_int identitiesPerMD = (INTS_PER_MD * BITS_PER_INT)/bValue;
  u_int checkSize = (maxID / BITS_PER_INT) +1;
  int errBuffSize = 1024;
  char* errBuff = new char[errBuffSize];
  int ATTACKSIZE = MAX_GPU_THREADS/2;
  u_char* dataBuff = new u_char[ATTACKSIZE*DATA_LENGTH];
  DIR* dir;
  struct dirent *ent;
  bool logExisted = false;
  const u_char* packet;
  srand(time(NULL));
  u_int seed = rand() % 100;
  u_char* dataBUff = new u_char[packetLimit * DATA_LENGTH];
  //create||open logfile
  char fileName[50] = {};
  sprintf(fileName, "./logs/MockAttack-%d_%d_%d.csv", 
    timeInfo->tm_year + 1900, timeInfo->tm_mon + 1, timeInfo->tm_mday);
  //check to see if the file existed, if not print the csv header for this log.
  struct stat fileInfo = {};
  if(stat(fileName, &fileInfo) == -1){
    fprintf(stderr, "file: %s does not exist, will create\n", fileName);
  } else {
    logExisted = true;
  } 
  logfile = fopen(fileName, "aw");
  if(logfile == NULL){
    fprintf(stderr, "could not open %s\n", fileName);
    return false;
  } 
  if(!logExisted){
    //print the csv heade
    fprintf(logfile, "test,bValue,tValue,packet limit, duration (ms),identities found,time stamp\n"); 
  }

  if((dir = opendir(dataDir)) != NULL){
    u_int gpuResult = 0;
    u_int* pool  = new u_int[checkSize];
    while((ent = readdir(dir)) != NULL){
      memset(pool, 0, checkSize*sizeof(u_int));
      char* pcapName = ent->d_name;
      if(strncmp(ent->d_name, ".", 1) == 0) continue;
      //create a pcap handle
      char absPath[100];
      sprintf(absPath, "%s%s", dataDir, ent->d_name);
      pcapH = pcap_open_offline(absPath, errBuff);
      if(pcapH == NULL){
        fprintf(stderr, "could not open pcapfile %s\n", absPath);
        return false;
      }
      //read packets from handle
      do{
        packet = pcap_next(pcapH, &header);
                //gather secrets and set them in them in the pool
        u_char* currData = new u_char[DATA_LENGTH];
        memset(currData, 0, DATA_LENGTH*sizeof(u_char));
        memcpy(currData, packet + DATA_POSITION, DATA_LENGTH);
        u_char* currMD = new u_char[MD_LENGTH];
        memset(currData,0,MD_LENGTH*sizeof(u_char));
        SHA1(currData, DATA_LENGTH, currMD);
        for(int y = 0; y < identitiesPerMD; y++){
          int z = getNthIdentifier((u_int*) currMD, y, bValue, MD_LENGTH);
          setBit(pool, z);
        }
        for(int w = 0; w < NUMBER_K_SECRETS; w++){
          memset(currMD, 0, MD_LENGTH*sizeof(u_char));
          memcpy(currMD,packet + (w*MD_LENGTH) + K_SECRET_POSITION, MD_LENGTH);
          for(int y = 0; y < identitiesPerMD; y++){
            int z = getNthIdentifier((u_int*) currMD, y, bValue, MD_LENGTH);
            setBit(pool, z);
          }
        }
        pktcnt++;
      }while(packet && pktcnt < packetLimit);
      //sendto HandleDataV2
      StopWatchInterface* timer = NULL;
      sdkCreateTimer(&timer);
      sdkStartTimer(&timer);
      bool repeat = true;
      u_char* tempDataP = dataBuff;
      while(repeat){

        for(int x = 0; x < ATTACKSIZE; x++){
          //create data point to hash 
          seed++;
          memcpy(tempDataP, (u_char*) &seed, DATA_LENGTH);
          tempDataP= tempDataP + DATA_LENGTH;
        }

        handleDataV2(dataBuff, DATA_LENGTH, ATTACKSIZE, pool, checkSize, bValue,identitiesPerMD, tValue, &gpuResult, NULL);
        if(gpuResult != 0) {
          printf("broken\n");
          repeat = false;
        } else {
          tempDataP = dataBuff;
        }
      }
      sdkStopTimer(&timer);
      fprintf(logfile,"%s,%d,%d,%d,%f,%u,%s",
          dataDir, bValue, tValue, packetLimit,sdkGetTimerValue(&timer), gpuResult, asctime(timeInfo));
      sdkDeleteTimer(&timer);
      pcap_close(pcapH);
    }
  }else{
    fprintf(stderr, "Could not open %s\n", dataDir);
    return false;
  }
  fclose(logfile);
  delete[] errBuff;
  delete[] dataBuff;
  return true;
}

void CudaKernelTestSuite::MockAttack1024(){
  char dir[] = "./data/t1024_80/";
  int start = 80;
  int bvl = 1024;
  int tvl = 3; 
  while(start > 0){
    TEST_ASSERT(mockAttackTemplate(dir, bvl, tvl, start));
    start -= 5;
  }
}

void CudaKernelTestSuite::MockAttack2048(){
  char dir[] = "./data/t2048_80/";
  int start = 80;
  int bvl = 2048;
  int tvl = 0; 
  while(start > 0){
    TEST_ASSERT(mockAttackTemplate(dir, bvl, tvl, start));
    start -= 5;
  }

}
void CudaKernelTestSuite::MockAttack4096(){
  char dir[] = "./data/t4096_80/";
   int start = 80;
  int bvl = 4096;
  int tvl = 0; 
  while(start > 0){
    TEST_ASSERT(mockAttackTemplate(dir, bvl, tvl, start));
    start -= 5;
  }
}
void CudaKernelTestSuite::MockAttack8192(){
  char dir[] = "./data/t8192_80/";
   int start = 80;
  int bvl = 8192;
  int tvl = 0; 
  while(start > 0){
    TEST_ASSERT(mockAttackTemplate(dir, bvl, tvl, start));
    start -= 5;
  }
}

int main(void){
  cudaFree(0);
  mkdir("./logs", S_IRWXU | S_IRWXG | S_IRWXO);
  printf("b");
  CudaKernelTestSuite cts;
  Test::TextOutput output(Test::TextOutput::Verbose);
  cts.run(output);
  cudaDeviceReset();
  return 0;
}
