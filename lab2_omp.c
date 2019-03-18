#include <malloc.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
int debug =0;
int debug2 = 0;

void calctranspose(int M, int N, float* D, float** D_T){
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            (*D_T)[M*i+j] = D[N*j+i];
        }
    }
}
float absfunc(float a, float b){
    if(a>b){
        return a-b;
    }else{
        return b-a;
    }
}

void printMatrix(int m, int n, float ** mat){
    printf("printing matrix\n");
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            printf("%.5f ",(*mat)[n*i+j]);
        }
        printf("\n");
    }
    printf("print over\n");
}
void printMatrixint(int m, int n, int ** mat){
    printf("printing matrix\n");
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            printf("%d ",(*mat)[n*i+j]);
        }
        printf("\n");
    }
    printf("print over\n");
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
int subtract(int adim1, int adim2, float * a, int bdim1,int bdim2, float * b, float ** c){
    if(adim1!=bdim1 || adim2!=bdim2){
        return -1;
    }
    for(int i=0;i<adim1;i++){
        for(int j=0;j<adim2;j++){
            (*c)[adim2*i+j] = a[adim2*i+j]-b[adim2*i+j];
        }
    }
    return 0;
}
float sumsquareelements(int M, int N, float *m){
    float temp=0.0;
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            temp += (m[N*i+j])*(m[N*i+j]);
        }
    }
    return temp;
}

float sumabsoelements(int M, int N, float *m ){
    float temp=0.0;
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            temp += absfunc(m[N*i+j],0.0);
        }
    }
    return temp;
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
        temp += ((*u)[M*i+j])*((*u)[M*i+j]);
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
            // (*r)[M*diffell+j]=tempproj;
            for(int i=0;i<M;i++){
                u[M*i+j] = u[M*i+j] - tempproj*(e[M*i+diffell]);
            }
        }
        float normuj= norm(M,&u,j);
        if(normuj==0){
            if(0==debug) printf("division by zero possible here\n");
            for(int i=0;i<M;i++){
                e[M*i+j]=0;
            }
        }else{
            for(int i=0;i<M;i++){
                e[M*i+j]=(1.0/normuj)*(u[M*i+j]);
            }
        }
        
        // (*r)[M*j+j]=proj(M,&a,j,&e,j);
        // (*r)[M*j+j]=normuj;

    }
    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            (*q)[M*i+j] = e[M*i+j];
            // if(i>j){
            //     (*r)[M*i+j]=0;
            // }
        }
    }
    if(0==debug){

        float * q_tm = (float *)malloc((sizeof(float) *M*M));
        calctranspose(M,M,*q,&q_tm);
        int statusmultiply = multiply(M,M,q_tm,M,M,a,r);
        printf("q is\n");
        printMatrix(M,M,q);
        printf("r is\n");
        printMatrix(M,M,r);
        float * qmulr = (float *)malloc(sizeof(float) * M *M);
        multiply(M,M,*q,M,M,*r,&qmulr);
        printf("q x r is\n");
        printMatrix(M,M,&qmulr);
        printf("original matrix is\n");
        printMatrix(M,M,&a);
        free(q_tm);
        free(qmulr);
    }
    free(u);
    free(e);
    return 0;

}


int qrmodifiedfactors(int M, float * a, float ** q, float ** r){
    float * v = (float *)malloc(sizeof(float) * M *M);
    // float * e = (float *)malloc(sizeof(float) * M *M);
    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            v[M*i+j]=a[M*i+j];
            (*r)[M*i+j]=0.0;
        }
    }
    for(int i=0;i<M;i++){
        float tempnorm = norm(M,&v,i);
        if(tempnorm==0){
            if(0==debug) printf("division by zero being done\n");
        }
        (*r)[M*i+i]= tempnorm;
        for(int rowiter=0;rowiter<M;rowiter++){
            (*q)[M*rowiter+i] = (1.0/tempnorm)*(v[M*rowiter+i]);
        }
        for(int j=i+1;j<M;j++){
            float rij = proj(M,q,i,&v,j);
            (*r)[M*i+j] = rij;
            for(int rowiter=0;rowiter<M;rowiter++){
                v[M*rowiter+j] = v[M*rowiter+j] - rij*((*q)[M*rowiter+i]);
            }

        }

    }
    free(v);
    // float * q_tm = (float *)malloc((sizeof(float) *M*M));
    // calctranspose(M,M,*q,&q_tm);
    // int statusmultiply = multiply(M,M,q_tm,M,M,a,r);
    if(0==debug){

        // printf("q is\n");
        // printMatrix(M,M,q);
        // printf("r is\n");
        // printMatrix(M,M,r);
        float * qmulr = (float *)malloc(sizeof(float) * M *M);
        multiply(M,M,*q,M,M,*r,&qmulr);
        // printf("q x r is\n");
        // printMatrix(M,M,&qmulr);
        // printf("original matrix is\n");
        // printMatrix(M,M,&a);
        // free(q_tm);
        float * diffm = (float *)malloc(sizeof(float) * M *M);
        subtract(M,M,a,M,M,qmulr,&diffm);
        float tempabsodiff = sumabsoelements(M,M,diffm);
        printf("Absolute diff qr is %.6f -------------------------------\n",tempabsodiff);
        if(tempabsodiff>0.001){
            printf("q is\n");
            printMatrix(M,M,q);
            printf("r is\n");
            printMatrix(M,M,r);
            printf("q x r is\n");
            printMatrix(M,M,&qmulr);
            printf("original matrix is\n");
            printMatrix(M,M,&a);
            printf("diff matrix is \n");
            printMatrix(M,M,&diffm);
        }
        free(diffm);
        free(qmulr);
    }
    return 0;

}


int findeigen(int M, float * darg, float ** eigenvector, float ** eigenvalues){
    if(debug==0) printf("original d is \n");
    if(debug==0) printMatrix(M,M,&darg);
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
        if(0==debug) printf("loop %d starting\n",numloop);
        statusqr = qrmodifiedfactors(M, d_eval, &qmat, &rmat);
        statusmultiply = multiply(M,M,rmat,M,M,qmat, &d_evalnew);
        statusmultiply = multiply(M,M,e_evec,M,M,qmat,&e_evecnew);
        if(debug==0) printf("D_eval is \n");
        if(debug==0) printMatrix(M,M,&d_eval);
        if(debug==0) printf("D_evalnew is \n");
        if(debug==0) printMatrix(M,M,&d_evalnew);
        if(debug==0) printf("e_eval is \n");
        if(debug==0) printMatrix(M,M,&e_evec);
        if(debug==0) printf("e_evalnew is \n");
        if(debug==0) printMatrix(M,M,&e_evecnew);
        // if(debug==0) printf("q is \n");
        // if(debug==0) printMatrix(M,M,&qmat);
        // if(debug==0) printf("r is \n");
        // if(debug==0) printMatrix(M,M,&rmat);

        numchangesd=0;
        numchangese=0;
        float tempdiff1=0.0;
        float tempdiff2=0.0;
        for(int i=0;i<M;i++){
            for(int j=0;j<M;j++){
                // if(absfunc(d_evalnew[M*i+j],d_eval[M*i+j])>0.0001){
                //     numchangesd+=1;
                // }
                // if(absfunc(e_evecnew[M*i+j],e_evec[M*i+j])>0.0001){
                //     numchangese+=1;
                // }
                tempdiff1 += absfunc(d_evalnew[M*i+j],d_eval[M*i+j]);
                tempdiff2 += absfunc(e_evecnew[M*i+j],e_evec[M*i+j]);
            }
        } 
        for(int i=0;i<M;i++){
            for(int j=0;j<M;j++){
                d_eval[M*i+j]=d_evalnew[M*i+j];
                e_evec[M*i+j]=e_evecnew[M*i+j];
            }
        } 
        numloop+=1;
        if(0==debug) printf("loop %d ending with numchangesd %d and numchangese %d\n",numloop, numchangesd, numchangese);
        
        if(0==debug2) printf("eigen %d loop with diff %.6f %.6f\n",numloop, tempdiff1, tempdiff2);
        if(tempdiff1 < 0.001 && tempdiff2<0.001){
            if(0==debug) printf("breaking on loop %d\n",numloop);
            if(0==debug2) printf("breaking on loop %d\n",numloop);
            break;
        }
        if(numloop>4){
            if(0==debug) printf("eigen end loop with diff %.6f %.6f\n", tempdiff1, tempdiff2);
            if(0==debug2) printf("eigen end loop with diff %.6f %.6f\n", tempdiff1, tempdiff2);
            break;
        }
    }
    if(0==debug) printf("D after convergence is\n");
    if(0==debug) printMatrix(M,M,&d_eval);
    if(0==debug) printf("E after convergence is \n");
    if(0==debug) printMatrix(M,M,&e_evec);
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
    if(0==debug) printf("Sort order is\n");
    if(0==debug) printMatrixint(1,M,&eigenvaluessortedorder);
    float * sigmamatrix = (float *)malloc(sizeof(float) * N*M);
    float * sigmainvmatrix = (float *)malloc(sizeof(float)*M*N);
    float * V = (float *)malloc(sizeof(float) * M *M);
    for(int i=0;i<M*N;i++){
        sigmainvmatrix[i]=0;
        sigmamatrix[i]=0;
    }
    for(int i=0;i<M*M;i++){
        V[i]=0;
    }
    for(int i=0;i<N;i++){
        float tempeigen = eigenvalues[eigenvaluessortedorder[i]];
        tempeigen = (float) sqrt(tempeigen);
        (*SIGMA)[i] = tempeigen;
        if(tempeigen==0){
            if(0==debug) printf("division by zero eigen possible here ============================\n");
        }
        sigmamatrix[M*i+i] = tempeigen;
        sigmainvmatrix[N*i+i] = 1.0/(tempeigen);
        for(int j=0;j<M;j++){
            V[M*j+i] = eigenvector[M*j+eigenvaluessortedorder[i]];
        }
    }
    calctranspose(M,M,V,V_T);
    float * tempmult = (float *)malloc(sizeof(float)*N*M);
    statusmultiply = multiply(N,M,D_T,M,M,V,&tempmult);
    statusmultiply = multiply(N,M,tempmult,M,N,sigmainvmatrix,U);
    if(0==debug){
        float * tempmult2 = (float *)malloc(sizeof(float)*N*M);
        statusmultiply = multiply(N,N,*U,N,M,sigmamatrix,&tempmult);
        statusmultiply = multiply(N,M,tempmult,M,M,*V_T,&tempmult2);
        printf("U is\n");
        printMatrix(N,N,U);
        printf("V_T is\n");
        printMatrix(M,M,V_T);
        printf("Sigma matrix is\n");
        printMatrix(N,M,&sigmamatrix);
        printf("Sigma inv is\n");
        printMatrix(M,N,&sigmainvmatrix);
        printf("Sigma is\n");
        printMatrix(1,N,SIGMA);
        printf("usigmavt is \n");
        printMatrix(N,M,&tempmult2);
        printf("ori m or d_t was");
        printMatrix(N,M,&D_T);
        printf("done svd\n");
    }
    if(0==debug2){
        float * tempmult3 = (float *)malloc(sizeof(float)*N*M);
        statusmultiply = multiply(N,N,*U,N,M,sigmamatrix,&tempmult);
        statusmultiply = multiply(N,M,tempmult,M,M,*V_T,&tempmult3);
        float * tempmult4 = (float *)malloc(sizeof(float)*N*M);
        int statussubtract = subtract(N,M,D_T,N,M,tempmult3,&tempmult4);
        float sumsquare= sumsquareelements(N,M,tempmult4);
        printf("Subtract Matrix is\n");
        printMatrix(N,M,&tempmult4);
        printf("sumsquare is %.6f after divided is %.6f\n ", sumsquare, sumsquare/(N*M));
        //tempmult3 is u*sigma*v_t
        
    }

}

// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
void PCA(int retention, int M, int N, float* D, float* U, float* SIGMA, float** D_HAT, int *K)
{
    float sigmasum = 0.0;
    for(int i=0;i<N;i++){
        sigmasum+= SIGMA[i];
    }
    float targetthressigma = retention*sigmasum/100.0;
    float tempsigmasum = 0.0;
    int k=0;
    for(int i=0;i<N;i++){
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
    if(0==debug){

        printf("D is\n");
        printMatrix(M,N,&D);
        printf("D_Hat is\n");
        printMatrix(M,k,D_HAT);
        printf("k is %d\n",k);
        printf("pca done\n");
    }
    if(0==debug2){
        printf("k is %d\n",k);
    }
}
