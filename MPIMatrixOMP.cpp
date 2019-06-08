
#include <stdio.h>
#include <sys/time.h>
#include <pthread.h>
#include <omp.h>
#include "mpi.h"
#define N 4
MPI_Status status;



double a[N][N],b[N][N],c[N][N];

int main(int argc, char **argv)
{
  int numtasks,taskid,numworkers,source,dest,rows,offset,i,j,k;

  struct timeval start, stop;
    
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  numworkers = numtasks-1;

  if (taskid == 0) {
    for (i=0; i<N; i++) {
      for (j=0; j<N; j++) {
        a[i][j]= 2.0;
        b[i][j]= 2.0;
      }
    }
      
     gettimeofday(&start, 0);
      
      
    /* send matrix data to the worker tasks */
    rows = N/numworkers;
    offset = 0;

    for (dest=1; dest<=numworkers; dest++)
    {
      MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
      MPI_Send(&a[offset][0], rows*N, MPI_DOUBLE,dest,1, MPI_COMM_WORLD);
      MPI_Send(&b, N*N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
      offset = offset + rows;
    }

    /* wait for results from all worker tasks */
    for (i=1; i<=numworkers; i++)
    {
      source = i;
      MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
      MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
      MPI_Recv(&c[offset][0], rows*N, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
    }
      
      
      gettimeofday(&stop, 0);
  
      printf("Here is the result matrix:\n");
    for (i=0; i<N; i++) {
      for (j=0; j<N; j++)
        printf("%6.2f   ", c[i][j]);
      printf ("\n");
    }
      
      printf("Time = %.6f\n\n",
              (stop.tv_sec+stop.tv_usec*1e-6)-(start.tv_sec+start.tv_usec*1e-6));
      
      
  }

  if (taskid > 0) {
    source = 0;
    MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&a, rows*N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&b, N*N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);

    /* Matrix multiplication */
    #pragma omp parallel num_threads(2)
    {
        #pragma omp for
        for (k=0; k<N; k++){
            for (i=0; i<rows; i++) {
                c[i][k] = 0.0;
                for (j=0; j<N; j++) {
                    c[i][k] += a[i][j] * b[j][k];}
        }
      }
    }


    MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    MPI_Send(&c, rows*N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
  }

  MPI_Finalize();

}
