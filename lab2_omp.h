#ifndef LAB2_OMP_H
#define LAB2_OMP_H


/*
To be implemented
Arguments:
    M : number of rows (samples) in input matrix D (input)
    N : number of columns (features) in input matrix D (input)
    D : 1D Array of M x N input matrix in row-major, (input)
        #elements in D is (M * N)
        --------------------------------------------------------------------------------------
        | D[0][0] | D[0][1] | ... | D[0][N-1] | D[1][0] | ... | D[1][N-1] | ... | D[M-1][N-1] |
        --------------------------------------------------------------------------------------
    U : 1D array of N x N real matrix in row-major (to be computed)
        --------------------------------------------------------------------------------------
        | U[0][0] | U[0][1] | ... | U[0][N-1] | U[1][0] | ... | U[1][N-1] | ... | U[N-1][N-1] |
        --------------------------------------------------------------------------------------
    SIGMA : 1D array of N x M diagonal matrix of positive real numbers (to be computed)
        format: consists only digonal elements
        #elements in SIGMA is N
        -------------------------------------------------------------------
        | SIGMA[0][0] | SIGMA[1][1] | SIGMA[2][2] | ... | SIGMA[N-1][N-1] |
        -------------------------------------------------------------------
    V_T : 1D array of M x M real matrix in row-major (to be computed)
        -------------------------------------------------------------------------------
        | V_T[0][0] | V_T[0][1] | ... | V_T[0][M-1] | V_T[1][0] | ... | V_T[M-1][M-1] |
        -------------------------------------------------------------------------------
*/
void SVD(int M, int N, float* D, float** U, float** SIGMA, float** V_T);

/*
To be implemented
Arguments:
    retention : percentage of information to be retained by PCA, ie
        retention = 90 means 90% of information should be retained
    M : number of rows (samples) in input matrix D (input)
    N : number of columns (features) in input matrix D (input)
    D : 1D Array of M x N input matrix in row-major, (input)
        #elements in D is (M * N)
        --------------------------------------------------------------------------------------
        | D[0][0] | D[0][1] | ... | D[0][N-1] | D[1][0] | ... | D[1][N-1] | ... | D[M-1][N-1] |
        --------------------------------------------------------------------------------------
    U : 1D array of N x N real matrix in row-major (input)
        --------------------------------------------------------------------------------------
        | U[0][0] | U[0][1] | ... | U[0][N-1] | U[1][0] | ... | U[1][N-1] | ... | U[N-1][N-1] |
        --------------------------------------------------------------------------------------
    SIGMA : 1D array of N x M diagonal matrix of positive real numbers (input)
        format: consists only digonal elements
        #elements in SIGMA is N
        -------------------------------------------------------------------
        | SIGMA[0][0] | SIGMA[1][1] | SIGMA[2][2] | ... | SIGMA[N-1][N-1] |
        -------------------------------------------------------------------
    D_HAT : 1D array of reduced M x K real matrix in row-major (to be computed)
        -----------------------------------------------------------------------------------------
        | D_HAT[0][0] | D_HAT[0][1] | ... | D_HAT[0][K-1] | D_HAT[1][0] | ... | D_HAT[M-1][K-1] |
        -----------------------------------------------------------------------------------------
    K : number of columns (features) in reduced matrix (to be computed)
*/
void PCA(int retention, int M, int N, float* D, float* U, float* SIGMA, float** D_HAT, int *K);

int qrfactors(int M, float * a, float ** q, float ** r);
int qrmodifiedfactors(int M, float * a, float ** q, float ** r);
#endif