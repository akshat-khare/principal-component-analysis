#include <malloc.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>

void calctranspose(int M, int N, float* D, float** D_T){
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            (*D_T)[M*i+j] = D[N*j+i];
        }
    }
}

int multiply(int adim1, int adim2, float * a, int bdim1,int bdim2, float * b, float ** c){
    if(adim2!=bdim1){
        return -1;
    }
    for(int i1=0;i1<adim1;i1++){
        for(int j1=0;j1<bdim2;j1++){
            float temp=0.0;
            for(int i2=0;i2<adim2;i2++){
                temp += (a[adim2*i1+i2])*(b[bdim2*i2+ j1]);
            }
            (*c)[bdim2*i1+j1]=temp;
        }
    }
    return 0;
}

float absfunc(float a, float b){
    if(a>b){
        return a-b;
    }else{
        return b-a;
    }
}
float proj(int M,float ** a, int j, float ** e, int k){
    float temp =0.0;
    for(int i=0;i<M;i++){
        temp += ((*a)[M*i+j])*((*e)[M*i+k]);
    }
    return temp;
}
float norm(int M, float ** u, int j){
    float temp=0.0;
    for(int i=0;i<M;i++){
        temp += ((*u)[M*i+j])*(((*u)[M*i+j]));
    }
    temp = (float) sqrt(temp);
    return temp;

}
int qrfactors(int M, float * a, float ** q, float ** r){
    float * u = (float *)malloc(sizeof(float) * M *M);
    float * e = (float *)malloc(sizeof(float) * M *M);
    for(int j=0;j<M;j++){
        for(int i=0;i<M;i++){
            u[M*i+j] = a[M*i+j];
        }
        for(int diffell=0; diffell<j;diffell++){
            float tempproj = proj(M,&a,j,&e,diffell);
            (*r)[M*diffell+j]=tempproj;
            for(int i=0;i<M;i++){
                u[M*i+j] = u[M*i+j] - tempproj*(e[M*i+diffell]);
            }
        }
        float normuj= norm(M,&u,j);
        if(normuj==0){
            printf("division by zero possible here-------------------------");
        }
        for(int i=0;i<M;i++){
            e[M*i+j]=(u[M*i+j])/(normuj);
        }
        (*r)[M*j+j]=proj(M,&a,j,&e,j);

    }
    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            (*q)[M*i+j] = e[M*i+j];
            if(i>j){
                (*r)[M*i+j]=0;
            }
        }
    }
    free(u);
    free(e);
    return 0;

}

int findeigen(int M, float * darg, float ** eigenvector, float ** eigenvalues){
    float * d_eval = (float *)malloc(sizeof(float) * M*M);
    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            d_eval[M*i+j]=darg[M*i+j];
        }
    }
    float * e_evec = (float *)malloc(sizeof(float) * M*M);
    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            if(i==j){
                e_evec[M*i+j]=1;
            }else{
                e_evec[M*i+j]=0;
            }
        }
    }
    float * d_evalnew = (float *)malloc(sizeof(float) * M*M);
    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            d_evalnew[M*i+j]=darg[M*i+j];
        }
    }
    float * e_evecnew = (float *)malloc(sizeof(float) * M*M);
    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            if(i==j){
                e_evecnew[M*i+j]=1;
            }else{
                e_evecnew[M*i+j]=0;
            }
        }
    }
    int numloop=0;
    float * qmat = (float *)malloc(sizeof(float) * M*M);
    float * rmat = (float *)malloc(sizeof(float) * M*M);
    int numchangesd=0;
    int numchangese=0;
    int statusqr;
    int statusmultiply;
    while(0==0){
        statusqr = qrfactors(M, d_eval, &qmat, &rmat);
        statusmultiply = multiply(M,M,rmat,M,M,qmat, &d_evalnew);
        statusmultiply = multiply(M,M,e_evec,M,M,qmat,&e_evecnew);
        numchangesd=0;
        numchangese=0;
        for(int i=0;i<M;i++){
            for(int j=0;j<M;j++){
                if(absfunc(d_evalnew[M*i+j],d_eval[M*i+j])>0.0001){
                    numchangesd+=1;
                }
                if(absfunc(e_evecnew[M*i+j],e_evec[M*i+j])>0.0001){
                    numchangese+=1;
                }
            }
        } 
        for(int i=0;i<M;i++){
            for(int j=0;j<M;j++){
                d_eval[M*i+j]=d_evalnew[M*i+j];
                e_evec[M*i+j]=e_evecnew[M*i+j];
            }
        } 
        numloop+=1;
        if(numchangesd<1 && numchangese<1){
            break;
        }
    }
    for(int i=0;i<M;i++){
        (*eigenvalues)[i]=d_eval[M*i+i];
        for(int j=0;j<M;j++){
            (*eigenvector)[M*i+j]=e_evec[M*i+j];
        }
    }
    return 0;

}
int customsort(int M, float * arr, int ** order){
    for(int i=0;i<M;i++){
        (*order)[i]=i;
    }
    int temp;
    for(int i=0;i<M-1;i++){
        for(int j=0;j<M-i-1;j++){
            if(arr[(*order)[j]]< arr[(*order)[j+1]]){
                temp = (*order)[j+1];
                (*order)[j+1]=(*order)[j];
                (*order)[j] = temp;
            }
        }
    }
    return 0;
}
// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void SVD(int M, int N, float* D, float** U, float** SIGMA, float** V_T)
{
    //Sigma is not in matrix form
    //First calculate d_t
    float * D_T = (float *)malloc(sizeof(float) * N*M);
    calctranspose(M, N, D, &D_T);
    //now we need to calculate svd of d_t
    //m of example is d_t
    //m_t of example is D

    //need to find m_t.m which is d.d_t
    float * d_multiply_d_t = (float *)malloc(sizeof(float) * M*M);
    int statusmultiply = multiply(M, N, D, N, M, D_T, &d_multiply_d_t);

    //need to find the eigen values of d_multiply_d_t
    float * eigenvector = (float *)malloc(sizeof(float) * M*M);
    float * eigenvalues = (float *)malloc((sizeof(float) * M));
    int statuseigen = findeigen(M, d_multiply_d_t, &eigenvector, &eigenvalues);
    int * eigenvaluessortedorder = (int *)malloc(sizeof(int)*M);
    int statussorted = customsort(M,eigenvalues,&eigenvaluessortedorder);
    float * sigmainvmatrix = (float *)malloc(sizeof(float)*M*M);
    float * V = (float *)malloc(sizeof(float) * M *M);
    for(int i=0;i<M*M;i++){
        sigmainvmatrix[i]=0;
    }
    for(int i=0;i<M;i++){
        (*SIGMA)[i] = eigenvalues[eigenvaluessortedorder[i]];
        sigmainvmatrix[M*i+i] = 1.0/(eigenvalues[eigenvaluessortedorder[i]]);
        for(int j=0;j<M;j++){
            V[M*j+i] = eigenvector[M*j+eigenvaluessortedorder[i]];
        }
    }
    calctranspose(M,M,V,V_T);
    float * tempmult = (float *)malloc(sizeof(float)*N*M);
    statusmultiply = multiply(N,M,D_T,M,M,V,&tempmult);
    statusmultiply = multiply(N,M,tempmult,M,M,sigmainvmatrix,U);


}

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void PCA(int retention, int M, int N, float* D, float* U, float* SIGMA, float** D_HAT, int *K)
{
    float sigmasum = 0.0;
    for(int i=0;i<M;i++){
        sigmasum+= SIGMA[i];
    }
    float targetthressigma = retention*sigmasum/100.0;
    float tempsigmasum = 0.0;
    int k=0;
    for(int i=0;i<M;i++){
        k+=1;
        tempsigmasum+=SIGMA[i];
        if(tempsigmasum>targetthressigma){
            break;
        }
    }
    *K=k;
    float * concatu = (float *)malloc(sizeof(float) * N*k);
    for(int i=0;i<N;i++){
        for(int j=0;j<k;j++){
            concatu[k*i+j] = U[N*i+j];
        }
    }
    *D_HAT = (float *)malloc(sizeof(float) * M*k);
    int statusmultiply = multiply(M, N, D, N, k, concatu, D_HAT);

}
