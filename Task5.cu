#include <iostream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>

using namespace std;

void help(){
	cout << "--accuracy <double> --size <int> --limit <int>" << endl; 
}

void error_message(){
	cout << "Arguments weren't parsed correctly. Send --help argument to get help" << endl;
}

__global__ void init(double* arr, int size){

	int k = size - 1;
	double step = (double)10/size;

	arr[0] = 10;
	arr[k] = 20;
	arr[k * size] = 20;
	arr[k * size + k] = 30;
	for(int i = 1; i < k; i++){
		arr[i] = arr[i - 1] + step;
		arr[k * size + i] = arr[k * size + (i - 1)] + step;
		arr[i * size] = arr[(i - 1) * size] + step;
		arr[i * size + k] = arr[(i - 1) * size + k] + step;
	}

	for(int i = 1; i < size - 1; i++){
		for(int j = 1; j < size - 1; j++) arr[i * size + j] = 0;
	}
}

int main(int argc, char* argv[]){
	double acc, loss = 1.0;
	int size, lim;
	double *arrprev, *arrnew, *arrloss;

	if(argc == 2 && argv[1] == "--help"){
		help();
		exit(0);
	}
	if(string(argv[1]) == "--accuracy") acc = atof(argv[2]);
	else{
		error_message();
		exit(0);
	}
	if(string(argv[3]) == "--size") size = atoi(argv[4]);
	else{
		error_message();
		exit(0);
	}
	if(string(argv[5]) == "--limit") lim = atoi(argv[6]);
	else{
		error_message();
		exit(0);
	}

	int rank, group_size;
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &group_size);

	cudaSetDevice(rank);

	cudaMalloc(&arrprev, sizeof(double) * (size * size));
	cudaMalloc(&arrnew, sizeof(double) * (size * size));
	cudaMalloc(&arrloss, sizeof(double) * (size * size));

	init<<<1, 1>>>(arrprev, size);
	init<<<1, 1>>>(arrnew, size);

	while(loss > acc && iter <= lim){

	}
}
