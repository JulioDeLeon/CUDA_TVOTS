#include "supplement.h"
using namespace std;

int 
main(int argc, char** argv){
  findCudaDevice(argc, (const char**) argv);
  cudaFree(0);
  char* fileName = NULL;
  opterr = 0;
  int c = 0;
  u_char* seed = new u_char[DATA_LENGTH];
  u_int* intSeed = (u_int*) seed;
  srand(time(NULL));
  intSeed[0] = rand();
  while((c = getopt(argc, argv,"f:b:t")) != -1){
    switch(c){
      case 'f':
        fileName = optarg;
        break;
      case '?':
        if(optopt == 'c')
          fprintf(stderr, "option -%c requires an argument.\n", optopt);
        else if (isprint(optopt))
          fprintf(stderr, "Unknown option '-%c'.\n", optopt);
        else
          fprintf(stderr, "Unknown option '\\x%x'.\n", optopt);

        return 1;
      default:
        abort();
    }
  }
  if(fileName == NULL){
    fprintf(stderr, "file name (-f [file name]) is required\n"); 
    exit(1);
  }else{
    printf("Using %s for pcap file\n", fileName);
  }
  pcap_t* test_handle;
  char* dev;
  //char test[5] = "eth0";
  struct pcap_pkthdr header;
  const u_char* packet;
  int pktcnt = 0;
  int secretsSeen = 0;
  int errBuffSize = 1024;
  double bValueLog = log2(1024);
  u_int bValue = static_cast<u_int>(bValueLog + 5);
  u_int tValue = 1;
  u_int maxID = (1<<bValue) - 1;
  u_int identitiesPerMD = (INTS_PER_MD * BITS_PER_INT)/bValue;
  u_int checkSize = ((maxID) / BITS_PER_INT) + 1;
  char* errBuff = new char[errBuffSize];
  u_char* dataBuff = new u_char[TARGET_NUMBER_PACKETS * DATA_LENGTH];
  u_char* secretBuff = new u_char[TARGET_NUMBER_PACKETS * (NUMBER_K_SECRETS) * K_SECRET_LENGTH];
  pcap_t* pcapH = pcap_open_offline(fileName, errBuff);
  if(pcapH == NULL){
    printf("Failed to open %s\n", fileName);
    exit(1);
  }
 
  /* Open the session in promiscuous mode 
  test_handle = pcap_open_live(test, BUFSIZ, 1, 1000, errbuff);
  if (test_handle == NULL) {
    fprintf(stderr, "Couldn't open device %s: %s\n", dev, errbuff);
    return(2);
  }
  */


  u_char* tempDataP = dataBuff;
  u_char* tempSecretP = secretBuff;
  do{
	packet = pcap_next(pcapH, &header);
	if(!packet || header.len != TARGET_PACKET_LENGTH) continue;
	TVOTS_Packet* temp = new TVOTS_Packet(packet);
        u_int* data = (u_int*) seed;
        data[0] += MAX_GPU_THREADS;
	memcpy(tempDataP, (u_char*)data, DATA_LENGTH);
	tempDataP = tempDataP + DATA_LENGTH;

	u_char* tempSSecret = temp->getSaltSecret();
	memcpy(tempSecretP, tempSSecret, SALT_SECRET_LENGTH);
	tempSecretP = tempSecretP + SALT_SECRET_LENGTH;
	secretsSeen++;
	for(int x = 0; x < NUMBER_K_SECRETS; x++){
		memcpy(tempSecretP, packet + K_SECRET_POSITION + (x*K_SECRET_LENGTH), K_SECRET_LENGTH);
		secretsSeen++;
	}
	pktcnt++;
  } while (packet && pktcnt < TARGET_NUMBER_PACKETS);

  printf("[%d packets captured, secrets Seen %d]\n", pktcnt, secretsSeen);
  
  u_int* pool = new u_int[checkSize];
  memset(pool, 0, checkSize*sizeof(u_int));
  u_int* gpuResult = new u_int[checkSize];
  memset(gpuResult, 0, checkSize*sizeof(u_int));
  u_char* currMD = NULL;
  for(int x = 0; x < secretsSeen; x++){
	  currMD = secretBuff + (x * MD_LENGTH);
	  for(int y = 0; y < identitiesPerMD; y++){
		  int z = getNthIdentifier((u_int*) currMD, y, bValue, MD_LENGTH);
		  setBit(pool, z);
	  }
  }
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  handleDataV2(dataBuff, DATA_LENGTH, TARGET_NUMBER_PACKETS, pool,checkSize,bValue,identitiesPerMD,tValue,gpuResult,NULL);
  sdkStopTimer(&timer);
  printf("[Processing time: %f (ms)]\n", sdkGetTimerValue(&timer));
  sdkDeleteTimer(&timer);
 /*
  for(int j = 0; j < checkSize; j++){
    printf("%d ", pool[j]);
  } 
  printf("\n");
  for(int j = 0; j < checkSize; j++){
    printf("%d ", gpuResult[j]);
  }
  printf("\n");
  setBit(gpuResult, 4000);
  for(int j = 0; j < checkSize; j++){
    printf("%d ", gpuResult[j]);
  }
  printf("\n");
*/
  delete[] seed;
  delete[] pool;
  delete[] gpuResult;
  delete[] dataBuff;
  delete[] secretBuff;
  delete[] errBuff;
  pcap_close(pcapH);
  //printf("[%s]\n", (succ)?"PASSED":"FAILED");
  checkCudaErrors(cudaDeviceReset());
  return 0;
}

