#include <iostream>
#include <vector>

template <typename T>
void vecPrint(std::vector<T> myVec){
    int length = myVec.size();
    bool printedEllipses = false;
    std::cout<<"[";
    for (int i = 0; i < length; i++){
        if (length < 10 || (i < 5 || i > length-5)){
            std::cout<<myVec[i]<<" ";
        }else{
            if (~printedEllipses){
                std::cout<<"...";
                printedEllipses = true;
            }
        }
    }
    std::cout<<"\b]";
}

int main(){

    std::vector<int> a;
    for (int i = 0; i < 11; i++){
        a.push_back(i);
    }

    vecPrint<int>(a);


    return 0;
}