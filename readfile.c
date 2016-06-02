
#include <stdio.h>
#include <stdlib.h>
const int nsec = 384;
int main() {

  int i;
  float *array;

  array = (float *) malloc(sizeof(float)*nsec);


  for (i = 0; i < nsec; i++) {
    array[i] = i;
  }
  for (i = 0; i < nsec; i++) {
      printf("%f\n", array[(i-1)%(nsec-1)] );
  }
  return 0;
}
