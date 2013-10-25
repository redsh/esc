OpenACC Jacobi Relaxation Exercise
=========

Let's focus on the part that actually runs jacobi relaxation. 

``` c
A    = (TYPE *)malloc(ORDER*ORDER*sizeof(TYPE));
b    = (TYPE *)malloc(ORDER*sizeof(TYPE));
xold = (TYPE *)malloc(ORDER*sizeof(TYPE));
xnew = (TYPE *)malloc(ORDER*sizeof(TYPE));

//init A, b, xold with random values
...

acc_set_device_num(0,acc_device_nvidia);

start_time = omp_get_wtime();
 
 //A. Iteration Loop
 while (conv > TOL && k<kMAX)
 {
 	//Loop 2
    for (i=0; i<ORDER; i++){
       xval = (TYPE)0.0;
       // 2b. j loop over all j not equal to i
       for(j=0;j<i;j++){
         xval += A[i*ORDER+j] * xold[j];
       }
       for(j=i+1;j<ORDER;j++){
         xval += A[i*ORDER+j] * xold[j];
      }
       xnew[i]= (b[i] - xval)/A[i*ORDER+i];
     }
     //Loop 3
     // 3. test convergence and increment iteration count
     conv = 0.0;
     for(i=0;i<ORDER;i++){
       dif = xnew[i]-xold[i];
       conv += dif*dif;
     }
     k++;
     //Loop 4
     // 4. Copy x for next round
     for(j=0;j<ORDER;j++)
       xold[j] = xnew[j];


   }
   
   run_time = omp_get_wtime() - start_time;
   
...
//solution check

```

Memory transfers and parallelization directives
===

Each iteration of the jacobi algorithm uses A, xold and b as input and produces new xold. xnew, instead, is used to check the convergence error later in the code, when the algorithm has finished. Hence the whole while loop ```takes as input A, b xold and outputs xnew``` (and conv).

We can therefore run the iterations on the GPU not communicating anything but the convergence residual conv to the host memory. We can instruct the PGI OpenACC compiler to do so by putting the following line just above the while loop:

``` c
#pragma omp data pcopyin(A[0:ORDER*ORDER],b[0:ORDER],xold[0:ORDER]) pcopyout(xnew[0:ORDER])
while (conv > TOL && k<kMAX)
{
...
```

Next, we instruct the compiler to parallelize jacobi loops (2,3,4) and map them on the GPU.


``` c
 	//Loop 2
 	#pragma acc parallel loop
    for (i=0; i<ORDER; i++){
       double xval = (TYPE)0.0;
       // 2b. j loop over all j not equal to i
       for(j=0;j<i;j++){
         xval += A[i*ORDER+j] * xold[j];
       }
       for(j=i+1;j<ORDER;j++){
         xval += A[i*ORDER+j] * xold[j];
      }
       xnew[i]= (b[i] - xval)/A[i*ORDER+i];
     }
     //Loop 3
     // 3. test convergence and increment iteration count
     conv = 0.0;
     #pragma acc parallel loop reduction(+:conv)
     for(i=0;i<ORDER;i++){
       dif = xnew[i]-xold[i];
       conv += dif*dif;
     }
     k++;
     //Loop 4
     // 4. Copy x for next round
     #pragma acc parallel loop
     for(j=0;j<ORDER;j++)
       xold[j] = xnew[j];

```

Next, we compile and execute:

``` sh
make jac_solv && ./jac_solv

```
and check the correctness of the result versus the serial version:



```
Order 10000 solver, ave error = 0.048705 and 29 iterations in 0.751852 seconds
```


Some useful diagnostics
===

Using the env macro:

``` sh
export PGI_ACC_TIME=1
./jac_solv
```

We can get a glimpse of what's going on at runtime. This is particulary useful to check:

- ```(83)``` That data is copied only once
- ```(86,101,109)``` The 3 kernels (loop 2,3,4) are launched 29 times and their grid/block sizes make sense
- The time requirements of the various parts:

``` 
Accelerator Kernel Timing data
jac_solv.c
  main  NVIDIA  devicenum=0
    time(us): 447,397
    83: data region reached 1 time
        83: data copyin reached 50 times
             device time(us): total=169,689 max=3,562 min=27 avg=3,393
        115: data copyout reached 1 time
             device time(us): total=57 max=57 min=57 avg=57
    86: compute region reached 29 times
        86: kernel launched 29 times
            grid: [10000]  block: [256]
             device time(us): total=276,688 max=9,558 min=9,530 avg=9,540
            elapsed time(us): total=282,957 max=11,924 min=9,544 avg=9,757
    101: compute region reached 29 times
        101: kernel launched 29 times
            grid: [40]  block: [256]
             device time(us): total=425 max=40 min=10 avg=14
            elapsed time(us): total=874 max=57 min=26 avg=30
        101: reduction kernel launched 29 times
            grid: [1]  block: [256]
             device time(us): total=261 max=12 min=8 avg=9
            elapsed time(us): total=679 max=29 min=22 avg=23
    109: compute region reached 29 times
        109: kernel launched 29 times
            grid: [40]  block: [256]
             device time(us): total=277 max=19 min=8 avg=9
            elapsed time(us): total=703 max=34 min=23 avg=24
```

In particular, the diagnostic

```
86: kernel launched 29 times
       grid: [10000]  block: [256]
```
hints us that the core of the algorithm (~ matrix vector multiplication - diagonal element) was mapped as follows:

- Each row of the matrix is assigned to a CUDA block (in fact, ORDER=10000)
- Each cuda block computes the row(a).b (minus diagonal) dot product (as reductions in shared memory?)

Compiling with something like (we just added ```keepgpu``` to ```-ta```):

``` sh
/opt/pgi/linux86-64/2013/bin/pgc++ -O3 -acc -fast -ta=nvidia,time,keepgpu -Minfo jac_solv.c frandom.c
```
, we can also verify our speculations in the CUDA source generated by the PGI compiler, which will be dumped to the file ```jac_solv.n001.gpu```.

If the parallelization did not happen that smoothly (in more complex cases it may not), we could have hinted the compiler to parallelize that way by explicitly assign the two nested jacobi iteration loops to ```gangs``` and ```workers```, and to perform a reduction on the inner  ```worker``` loop.
Then, we can compare ```jac_solv.n001.gpu``` to see if we see any difference!


``` c
     #pragma acc parallel loop gang
     for (i=0; i<ORDER; i++){
       TYPE xval = (TYPE)0.0;
       // 2b. j loop over all j not equal to i
       
       #pragma acc loop worker reduction(+:xval)
       for(j=0;j<i;j++){
         xval += A[i*ORDER+j] * xold[j];
       }
       #pragma acc loop worker reduction(+:xval)
       for(j=i+1;j<ORDER;j++){
         xval += A[i*ORDER+j] * xold[j];
         
       xnew[i]= (b[i] - xval)/A[i*ORDER+i];
     }
```

Performance evaluation
====

What about the performance we gained adding 3 lines of code?
We can get a rough estimate of the single core baseline performance, compiling and executing a CPU, serial version with:
``` sh
g++ -O3 jac_solv.c frandom.c -lgomp -o jac_solv_cpu && ./jac_solv_cpu
```
which gives (Note: pgc++ does a better job because it vectorizes loops on the CPU):

```
Order 10000 solver, ave error = 0.048705 and 29 iterations in 3.954635 seconds
```

Whereas, the OpenACC GPU version
```sh 
/opt/pgi/linux86-64/2013/bin/pgc++ -g -O3 -acc -ta=nvidia,time,keepgpu -Minfo -DAPPLE  jac_solv.c  frandom.c  -o jac_solv_gpu && ./jac_solv_gpu
```
gives:

```
Order 10000 solver, ave error = 0.048705 and 29 iterations in 0.751852 seconds
```


