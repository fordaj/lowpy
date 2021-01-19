#include <thrust/host_vector.h> 
#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h> 
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <cstdlib>


struct dense{
    thrust::device_vector<double> x,w,b,y,z,tmp;
    int I,J;
    dense(int number_of_inputs, int number_of_outputs){
        I = number_of_inputs;
        J = number_of_outputs;
        x.resize(I);
        thrust::fill(x.begin(), x.end(), 3);
        tmp.resize(I);
        thrust::fill(tmp.begin(), tmp.end(), 0);
        w.resize(J*I);
        thrust::fill(w.begin(), w.end(), 1);
        b.resize(J);
        thrust::fill(b.begin(), b.end(), 1);
        y.resize(I);
        thrust::fill(y.begin(), y.end(), 1);
        z.resize(I);
        thrust::fill(z.begin(), z.end(), 1);
    }
    void propagate(){
        thrust::transform(w.begin(), w.begin()+I, x.begin(), tmp.begin(), thrust::multiplies<float>());
        J = thrust::transform_reduce(w.begin(), w.begin()+I, )
        for (int i = 0; i < I; i++){
            std::cout<<tmp[i]<<std::endl;
        }
        y[0] = thrust::reduce(tmp.begin(),tmp.end())
    }
};

int main(void){
    /*
    // generate 32M random numbers on the host
    thrust::host_vector<int> h_vec(32 << 20); 
    thrust::generate(h_vec.begin(), h_vec.end(), rand);
    // transfer data to the device
    thrust::device_vector<int> d_vec = h_vec; // sort data on the device
    thrust::sort(d_vec.begin(), d_vec.end()); // transfer data back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin()); 
    */
    int n = 100
    thrust::device_vector<double>x,w;
    x.resize(n);
    w.resize(n);
    thrust::fill(x.begin(), x.end(), 1);
    thrust::fill(w.begin(), w.end(), 4);
    int y = thrust::transform_reduce()

    dense layer(10,5);
    layer.propagate();

    return 0;
}
 