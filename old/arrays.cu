#include <stdio.h>
#include <iostream>
#include <thrust/host_vector.h> 
#include <thrust/device_vector.h> 
#include <ctime>
// Compile with:
// nvcc -o example example.cu
#define N 1000



double uniformRandom(){
    return static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
}

__global__ void cudaPropagate(const int number_of_inputs, const double *x, const double *w, const double *b, double *y, double *z) {
    int j = blockIdx.x, I = number_of_inputs;
    double sum = 0;
    for (int i = 0; i < I; i++){
        sum += w[i+I*j] * x[i];
        
    }
    y[j] = sum + b[j];
    z[j] = 1/(1+exp(-1*y[j]));
}


struct dense{
    int I, J;
    double *x_h, *x_d, *w_d, *w_h, *b_h, *b_d, *y_h, *y_d, *z_h, *z_d;
    
    dense(const int numInputs, const int numOutputs){
        I = numInputs;
        J = numOutputs;
        x_h = new double[I];
        cudaMalloc((void **)&x_d, I*sizeof(double));
        w_h = new double[J*I];
        cudaMalloc((void **)&w_d, J*I*sizeof(double));
        b_h = new double[J];
        cudaMalloc((void **)&b_d, J*sizeof(double));
        y_h = new double[J];
        cudaMalloc((void **)&y_d, J*sizeof(double));
        z_h = new double[J];
        cudaMalloc((void **)&z_d, J*sizeof(double));
        for (int i = 0; i < I; i++){
            x_h[i] = 1;
        }
        for (int j = 0; j < J; j++){
            b_h[j] = 1;
            y_h[j] = 0;
            z_h[j] = 0;
            for (int i = 0; i < I; i++){
                w_h[i+I*j] = 1;
            }
        }
        cudaMemcpy(x_d, x_h, I*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(w_d, w_h, J*I*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(b_d, b_h, J*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(y_d, y_h, J*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(z_d, z_h, J*sizeof(double), cudaMemcpyHostToDevice);
        
    }
    void propagate(double *x_d){
        cudaPropagate<<<J,1>>>(I, x_d, w_d, b_d, y_d, z_d);
    }
    void deviceToHost(){
        cudaMemcpy(x_h, x_d, I*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(w_h, w_d, J*I*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(b_h, b_d, J*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(y_h, y_d, J*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(z_h, z_d, J*sizeof(double), cudaMemcpyDeviceToHost);
    }
    void freeDevice(){
        cudaFree(x_d);
        cudaFree(w_d);
        cudaFree(b_d);
        cudaFree(y_d);
        cudaFree(z_d);
    }
    
    void printOut(){
        cudaMemcpy(x_h, x_d, I*sizeof(double), cudaMemcpyDeviceToHost);
        std::cout<<"x:[";
        for (int i = 0; i < I; i++){
            std::cout<<x_h[i]<<" ";
        }
        std::cout<<"\b]"<<std::endl;
    }

};

struct network{
    double alpha;
    thrust::device_vector<dense> layer;
    double *x_h, *x_d;
    network(double learning_rate){
        alpha = learning_rate;
        x_h = new double[784];
        cudaMemcpy(x_d, x_h, 784*sizeof(double), cudaMemcpyHostToDevice);
    }
    void add(dense newDenseObject){
        layer.push_back(newDenseObject);
    }
    void propagate(){
        for (int l = 0; l < layer.size(); l++){
            layer[l].propagate()
        }
    }
};

int main() {
    srand (static_cast <unsigned> (time(0)));
    dense l1(784,533);
    dense l2(533,533);
    dense l3(533,10);
    for (int i = 0; i < 60000; i++){
        l1.propagate();
        l2.propagate();
        l3.propagate();
        l1.propagate();
        l2.propagate();
        l3.propagate();
        l1.propagate();
        l2.propagate();
        l3.propagate();
        std::cout<<i<<std::endl;
    }
    l1.deviceToHost();
    l1.freeDevice();

    return 0;
}