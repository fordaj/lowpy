#include <stdio.h>

__global__ void propagate(
    const   int                     I,
    const   int                     iteration,
    const   double * __restrict__   x,
    const   double * __restrict__   w,
    const   double * __restrict__   b,
            double * __restrict__   y,
            double * __restrict__   z
){
    int j = blockIdx.x;
    double sum = 0;
    int inputIdx = 0;
    if (iteration > -1){
        inputIdx = iteration * I;
    }
    for (int i = 0; i < I; i++) sum += x[inputIdx+i]*w[j*I+i];
    y[j] = sum + b[j];
    z[j] = 1/(1 + exp(-y[j]));
}
__global__ void backpropagate(
    const   int                     iteration,
    const   int                     label,
            double * __restrict__   dedz,
    const   double * __restrict__   z,
    const   int                     J_n,
            double * __restrict__   w_n,
    const   double * __restrict__   dedz_n,
    const   double * __restrict__   dzdy_n,
            double * __restrict__   dzdy,
    const   double                  alpha,
    const   int                     I,
            double * __restrict__   b,
            double * __restrict__   w,
    const   double * __restrict__   x,
    const   double                  beta,
            double * __restrict__   vtb,
            double * __restrict__   vtw
){
    int j   = blockIdx.x;
    int I_n = gridDim.x;
    int inputIdx = 0;
    if (iteration > -1){
        inputIdx = iteration * I;
    }
    if (label > -1){
        if (j == label){
            dedz[j] = z[j] - 1;
        }else{
            dedz[j] = z[j] - 0;
        }
    }else{
        double sum = 0;
        for (int j_n = 0; j_n < J_n; j_n++) sum += w_n[j+j_n*I_n] * dedz_n[j_n] * dzdy_n[j_n];
        dedz[j] = sum;
    }
    dzdy[j] = z[j] * (1 - z[j]);
    b[j]   -= (beta * vtb[j] + alpha * dedz[j] * dzdy[j]);
    vtb[j]  = beta * vtb[j] + alpha * dedz[j] * dzdy[j];

    for (int i = 0; i < I; i++){
        w[j*I+i]    -= (beta * vtw[j*I+i] + alpha * dedz[j] * dzdy[j] * x[inputIdx+i]);
        vtw[j*I+i]  = beta * vtw[j*I+i] + alpha * dedz[j] * dzdy[j] * x[inputIdx+i];
    }
}
__global__ void argmax(
        const   int                     label,
        const   double * __restrict__   z,
                double *__restrict__    hits
){
    int j = blockIdx.x;
    int J = gridDim.x;
    if (j == 0){
        double maxVal = 0;
        int maxIdx = 0;
        for (int i = 0; i < J; i++){
            if (z[i] > maxVal){
                maxIdx = i;
                maxVal = z[i];
            }
        }
        if (maxIdx == label){
            hits[0] += 1;
        }
        hits[1] = maxIdx;
        hits[2] = maxVal;
    }
}