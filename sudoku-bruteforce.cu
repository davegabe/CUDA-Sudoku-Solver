#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

// declare C include here
extern "C" {
#include "flatten.c"
}

__device__ int isSolved;

// Print the sudoku.
void printSudoku(int *sudoku, int n) {
  int sqrtN = sqrt(n);
  printf(
      "#############################\nSUDOKU\n#############################\n");

  for (int i = 0; i < n; ++i) {
    if (i % sqrtN == 0) {
      printf("----------------------\n");
    }
    for (int j = 0; j < n; ++j) {
      if (j % sqrtN == 0) {
        printf("|");
      }
      if (sudoku[i * n + j] <= 15) {
        printf("%X ", sudoku[i * n + j]);
      } else {
        char c = 'F' + (sudoku[i * n + j] - 15);
        printf("%c ", c);
      }
    }
    printf("|\n");
    if (i == n - 1) {
      printf("----------------------\n\n");
    }
  }
}

// Read the sudoku from file.
int *readSudoku(const char *file, int *n) {
  FILE *fp = fopen(file, "r");
  int *sudoku = NULL;
  fscanf(fp, "%d", n);
  int tot = *n * *n, i;
  sudoku = (int *)malloc(tot * sizeof(int));
  for (i = 0; i < tot; ++i) {
    fscanf(fp, "%d", sudoku + i);
  }
  return sudoku;
}

// Check if the sudoku is solved.
__device__ int isSudokuSolved(int *sudoku, int n) {
  int sqrtN = sqrt((float)n);
  int i, j, k, l;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      if (sudoku[i * n + j] == 0) {
        return 0;
      }
      for (k = 0; k < n; ++k) {
        if (k != j && sudoku[i * n + j] == sudoku[i * n + k]) {
          return 0;
        }
      }
      for (k = 0; k < n; ++k) {
        if (k != i && sudoku[i * n + j] == sudoku[k * n + j]) {
          return 0;
        }
      }
      for (k = i / sqrtN * sqrtN; k < i / sqrtN * sqrtN + sqrtN; ++k) {
        for (l = j / sqrtN * sqrtN; l < j / sqrtN * sqrtN + sqrtN; ++l) {
          if (k != i && l != j && sudoku[i * n + j] == sudoku[k * n + l]) {
            return 0;
          }
        }
      }
    }
  }
  return 1;
}

// Recursion on device.
__device__ int *recursionBruteforce(int *sudoku, int n, int id) {
  // if sudoku has been solved already, return
  if (isSolved == 1) {
    return NULL;
  }
  // if is solved, notfy all threads and return
  if (isSudokuSolved(sudoku, n) == 1) {
    isSolved = 1;
    return sudoku;
  }

  // get the row and column of the cell
  int row = id / n, col = id % n;

  // if the cell is empty, try all possible values
  if (sudoku[row * n + col] == 0) {
    for (int i = 1; i <= n; ++i) {
      sudoku[row * n + col] = i;
      int *sudoku_copy;
      cudaMalloc((void **)&sudoku_copy, n * n * sizeof(int));
      for (int j = 0; j < n * n; ++j) {
        sudoku_copy[j] = sudoku[j];
      }
      int *solution = recursionBruteforce(sudoku_copy, n, id);
      if (solution != NULL) {
        // cudaFree(sudoku_copy);
        return solution;
      }
      // cudaFree(solution);
    }
  }
  return NULL;
}

// Kernel to call recursion on device.
__global__ void bruteforce(int *sudoku, int n, int *solution) {
  // Get the thread id
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("%d\n", id);
  int *result = recursionBruteforce(sudoku, n, id);
  if (result != NULL) {
    solution = result;
  }
}

int main(int argc, char *argv[]) {
  int n = 0;
  int *sudoku = readSudoku(argv[1], &n);
  int expand = (int)atoi(argv[2]);
  int blockSize = n;

  int isSolved_h = 0;
  cudaMemcpyToSymbol("isSolved", &isSolved_h, sizeof(int));

  int *sudokuTry = NULL, *solution = NULL;
  cudaMalloc((void **)&sudokuTry, n * n * expand * sizeof(int));
  cudaMemcpy(sudokuTry, sudoku, n * n * sizeof(int), cudaMemcpyHostToDevice);

  bruteforce<<<expand, n * n>>>(sudokuTry, n, solution);
  cudaDeviceSynchronize();

  // copy and print isSolved from device to host
  cudaMemcpy(&isSolved_h, &isSolved, sizeof(int), cudaMemcpyDeviceToHost);
  printf("isSolved: %d\n", isSolved_h);

  // copy and print solution from device to host
  int *solution_h = NULL;
  cudaMemcpy(&solution_h, &solution, sizeof(int), cudaMemcpyDeviceToHost);
  if (solution != NULL) {
    printSudoku(solution, n);
  } else {
    printf("Sudoku not solved!\n");
  }

  free(sudoku);
  return 0;
}