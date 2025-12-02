#include <stdio.h>
#include <assert.h>

inline
cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

const int TILE_DIM = 32;        // 数据分片大小
const int BLOCK_ROWS = 8;       // 
const int NUM_REPS = 100;       // 重复次数

// Check errors and print GB/s
void postprocess(const float *ref, const float *res, int n, float ms) {
    bool passed = true;
    for (int i = 0; i < n; i++) {
        if(ref[i] != res[i]) {
            printf("%d %f %f", i, ref[i], res[i]);
            printf("%25s\n", "*** FAILED ***");
            passed = false;
            break;
        }
    }
    if (passed)
        printf("%20.2f\n", 2 * n * sizeof(float) * NUM_REPS * 1e-6 / ms);
}

// simple copy kernel
// [例]：block(0,1) thread(0,0)
__global__ void copy(float *odata, const float *idata) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;    // 数据：x 偏移 [例]0*32 + 0
    int y = blockIdx.y * TILE_DIM + threadIdx.y;    // 数据：y 偏移 [例]1*32 + 0
    int width = gridDim.x * TILE_DIM;               // x轴 数据量 (nx) 32*32 = 1024
    
    // 一个线程处理 TILE_DIM / BLOCK_ROWS = 4 个数据
    // j += BLOCK_ROWS 数据按 BLOCK_ROWS 分隔
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) 
        odata[(y+j)*width + x] = idata[(y+j)*width + x];
}

// copy kernel using shared memory
// [例]：block(1,1) thread(0,0)
__global__ void copySharedMem(float *odata, const float *idata) {
    __shared__ float tile[TILE_DIM * TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;        // 数据: x偏移 [例]1*32 + 0
    int y = blockIdx.y * TILE_DIM + threadIdx.y;        // 数据: y偏移 [例]1*32 + 0
    int width = gridDim.x * TILE_DIM;                   // x轴向数据量 (nx) 32*32 = 1024

    // 一个线程处理 TILE_DIM / BLOCK_ROWS = 4 个数据
    // [例]idata[(32 + 0/8/16/24) * 1024 + 32] -> tile[(0/8/16/24) * 32 + 0]
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) 
        tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x] = idata[(y+j)*width + x];

    __syncthreads();

    // [例] tile[(0/8/16/24) * 32 + 0] -> odata[(32 + 0/8/16/24) * 1024 + 32]
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y+j)*width + x] = tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x];
}

// native transpose
// Global memory reads are coalesced but writes are not.
// [例]：block(1,0) thread(0,0)
__global__ void transposeNaive(float *odata, const float *idata) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;        // 数据: x偏移 [例]1*32 + 0
    int y = blockIdx.y * TILE_DIM + threadIdx.y;        // 数据: y偏移 [例]0*32 + 0
    int width = gridDim.x * TILE_DIM;                   // 如果nx != ny???

    // idata[(0 + 0/8/16/24) * 1024 + 32] -> odata[32*1024 + (0/8/16/24)]
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) 
        odata[x*width + (y + j)] = idata[(y+j)*width + x];
}

// coalesced transpose
// Uses shared memory to achieve coalesing in both reads and writes
// Tile width == #banks causes shared memory bank conflicts.
// [例]：block(1,0) thread(0,0) (x, y)
__global__ void transposeCoalesced(float *odata, const float *idata){
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;    // 数据: x偏移 [例]1*32 + 0
    int y = blockIdx.y * TILE_DIM + threadIdx.y;    // 数据: y偏移 [例]0*32 + 0
    int width = gridDim.x * TILE_DIM;

    // idata[(0/8/16/24)*1024 + 32] -> tile[(0/8/16/24)][0]
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = idata[(y+j) * width + x];

    __syncthreads();

    // transpose block offset
    // 保证沿组件变化最快的方向(threadIdx.x)内存连续, 
    // 即global访存格式为odata[f(threadIdx.y)*width + g(threadIdx.x)]
    // 原先x, y不适用, 原数据 block(1,0) 需要映射至 输出数据 block(0, 1) 对应blockIdx.y/x 互换
    x = blockIdx.y * TILE_DIM + threadIdx.x;        // 0*32 + 0
    y = blockIdx.x * TILE_DIM + threadIdx.y;        // 1*32 + 0

    // tile[0][(0/8/16/24)] -> odata[(32 + 0/8/16/24)*1024 + 0]
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) 
        odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

// No bank-conflict transpose
// Same as transposeCoalesced except the first tile dimension is padded 
// to avoid shared memory bank conflicts
__global__ void transposeNoBankConflicts(float *odata, const float *idata){
    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;    // 数据: x偏移 [例]1*32 + 0
    int y = blockIdx.y * TILE_DIM + threadIdx.y;    // 数据: y偏移 [例]1*32 + 0
    int width = gridDim.x * TILE_DIM;

    // idata[(32 + 0/8/16/24)*1024 + 32] -> tile[(0/8/16/24)][0]
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = idata[(y+j) * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // tile[0][(0/8/16/24)] -> odata[(32 + 0/8/16/24)*1024 + 32]
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}


/*
                  Routine         Bandwidth (GB/s)
                     copy              752.81
       shared memory copy              838.40
         native transpose              235.21
      coalesced transpose              580.99
  conflict-free transpose              836.77
*/

int main(int argc, char **argv) {
    const int nx = 1024;
    const int ny = 1024;
    const int mem_size = nx*ny*sizeof(float);

    dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    int devId = 0;
    if (argc > 1) devId = atoi(argv[1]);        // 获取设备
    
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, devId));
    printf("\nDevice : %s\n", prop.name);
    printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n",
           nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
    printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
           dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
    
    checkCuda(cudaSetDevice(devId));

    float *h_idata = (float*)malloc(mem_size);  // host input
    float *h_cdata = (float*)malloc(mem_size);  // host copy
    float *h_tdata = (float*)malloc(mem_size);  // host transpose
    float *gold    = (float*)malloc(mem_size);

    float *d_idata, *d_cdata, *d_tdata;
    checkCuda(cudaMalloc(&d_idata, mem_size));
    checkCuda(cudaMalloc(&d_cdata, mem_size));
    checkCuda(cudaMalloc(&d_tdata, mem_size));

    // check parameters && calculate execution configuration
    if (nx % TILE_DIM || ny % TILE_DIM) {
        printf("nx and ny must be a multiple of TILE_DIM\n");
        goto error_exit;
    }

    if (TILE_DIM % BLOCK_ROWS) {
        printf("TILE_DIM must be a multiple of BLOCK_ROWS\n");
        goto error_exit;
    }

    // host
    for (int j = 0; j < ny; j++) 
        for (int i = 0; i < nx; i++)
            h_idata[j*nx + i] = j*nx + i;
    
    // correct result for error checking
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
            gold[j*nx + i] = h_idata[i*nx + j];

    // host -> device
    checkCuda(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));
    float ms;

    // ----------------
    // time kernels
    // ----------------
    printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");

    // ----------------
    // copy
    // ----------------
    printf("%25s", "copy");
    checkCuda(cudaMemset(d_cdata, 0, mem_size));
    // warm up
    copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; i++) 
        copy<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
    postprocess(h_idata, h_cdata, nx*ny, ms);

    // ----------------
    // copySharedMem 
    // ----------------
    printf("%25s", "shared memory copy");
    checkCuda(cudaMemset(d_cdata, 0, mem_size));
    // warm up
    copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; i++) 
        copySharedMem<<<dimGrid, dimBlock>>>(d_cdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_cdata, d_cdata, mem_size, cudaMemcpyDeviceToHost));
    postprocess(h_idata, h_cdata, nx*ny, ms);

    // --------------
    // transposeNaive 
    // --------------
    printf("%25s", "native transpose");
    checkCuda(cudaMemset(d_tdata, 0, mem_size));
    // warm up
    transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; i++) 
        transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
    postprocess(gold, h_tdata, nx*ny, ms);

    // ------------------
    // transposeCoalesced 
    // ------------------
    printf("%25s", "coalesced transpose");
    checkCuda(cudaMemset(d_tdata, 0, mem_size));
    // warm up
    transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; i++)
        transposeCoalesced<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
    postprocess(gold, h_tdata, nx*ny, ms);

    // ------------------------
    // transposeNoBankConflicts
    // ------------------------
    printf("%25s", "conflict-free transpose");
    checkCuda(cudaMemset(d_tdata, 0, mem_size));
    // warm up
    transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUM_REPS; i++)
        transposeNoBankConflicts<<<dimGrid, dimBlock>>>(d_tdata, d_idata);
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    checkCuda(cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost));
    postprocess(gold, h_tdata, nx*ny, ms);


error_exit:
    // cleanup
    checkCuda(cudaFree(d_tdata));
    checkCuda(cudaFree(d_cdata));
    checkCuda(cudaFree(d_idata));
    free(h_idata);
    free(h_tdata);
    free(h_cdata);
    free(gold);
}

