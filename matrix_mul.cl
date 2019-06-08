__kernel void multiply_matrices(const int M, const int N, const int K,
                      const __global int* A,
                      const __global int* B,
                      __global int* C) {
    
    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)
 
    //printf("(%d,%d)\n ", globalRow, globalCol);
    // Compute a single element (loop over K)
    int acc = 0.0f;
    for (int k=0; k<K; k++) {
        acc += B[k*M + globalRow] * A[globalCol*K + k];
     //   printf("(%d,%d), values = (%d, %d)\n ", k*M + globalRow, globalCol*K + k, A[k*M + globalRow] , B[globalCol*K + k]);
    }
 
    // Store the result
    //printf("(%d,%d), loc = %d, value = %d\n ", globalRow, globalCol, globalCol*M + globalRow, acc);
    C[globalCol*M + globalRow] = acc;
}