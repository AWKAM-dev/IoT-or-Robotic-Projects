#include <stdio.h>
#include <math.h>

#define time_int 5

void riemannSums(float* input[]){
    float sum = 0;
    for(int i = 0; i < sizeof(input); i++){
        if(input[i] == NULL){
            break;
        }
        float delta_x = input[i+1] - input[i];
        sum += delta_x * time_int;
    }
    printf("Riemann Sum: %f\n", sum);
}

int main(){

}