#include <stdio.h>
#include <math.h>

#define time_int 5

float riemannSums(float* input[]){
    float sum = 0;
    for(int i = 0; i < sizeof(input); i++){
        if(input[i] == NULL){
            break;
        }
        float delta_x = input[i+1] - input[i];
        sum += delta_x * time_int;
    }
    printf("Riemann Sum: %f\n", sum);
    return sum;
}

float trapezoidalRule(float* input[]){

    float sum = 0;
    float h = (input[sizeof(input)-1] - input[0]) / (sizeof(input) - 1);
    float firstPoint = *input[0];
    float lastPoint = *input[sizeof(input)-1];

    for(int i = 0; i < sizeof(input); i++){
        sum += *input[i];
    }
    sum = (h / 2) * (firstPoint + lastPoint + (2 * sum));
    printf("Trapezoidal Rule: %f\n", sum);
    return sum;
}



int main(){

}