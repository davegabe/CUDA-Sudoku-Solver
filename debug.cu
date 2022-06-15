#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>

// Print the sudoku kernel.
__global__ void printSudokuDeviceKernel(int *sudoku, int sqrtN)
{
  int n = sqrtN * sqrtN;
  printf(
      "#############################\nSUDOKU\n#############################\n");

  for (int i = 0; i < n; ++i)
  {
    if (i % sqrtN == 0)
    {
      printf("----------------------\n");
    }
    for (int j = 0; j < n; ++j)
    {
      if (j % sqrtN == 0)
      {
        printf("|");
      }
      if (sudoku[i * n + j] <= 15)
      {
        printf("%X ", sudoku[i * n + j]);
      }
      else
      {
        char c = 'F' + (sudoku[i * n + j] - 15);
        printf("%c ", c);
      }
    }
    printf("|\n");
    if (i == n - 1)
    {
      printf("----------------------\n\n");
    }
  }
}

// Print the sudoku on device.
__device__ void printSudokuDevice(int *sudoku, int sqrtN)
{
  int n = sqrtN * sqrtN;
  printf(
      "#############################\nSUDOKU\n#############################\n");

  for (int i = 0; i < n; ++i)
  {
    if (i % sqrtN == 0)
    {
      printf("----------------------\n");
    }
    for (int j = 0; j < n; ++j)
    {
      if (j % sqrtN == 0)
      {
        printf("|");
      }
      if (sudoku[i * n + j] <= 15)
      {
        printf("%X ", sudoku[i * n + j]);
      }
      else
      {
        char c = 'F' + (sudoku[i * n + j] - 15);
        printf("%c ", c);
      }
    }
    printf("|\n");
    if (i == n - 1)
    {
      printf("----------------------\n\n");
    }
  }
}

// Print the expanded sudoku.
void printExpandedSudoku(int *sudoku, int n, int expand)
{
  int sqrtN = sqrt(n);
  printf("#############################\nEXPANDED SUDOKU\n#############################\n");
  for (int k = 0; k < expand; ++k)
  {
    printf(
        "#############################\nSUDOKU %d\n#############################\n", k);
    int currentSudoku = k * n * n;
    for (int i = 0; i < n; ++i)
    {
      if (i % sqrtN == 0)
      {
        printf("----------------------\n");
      }
      for (int j = 0; j < n; ++j)
      {
        if (j % sqrtN == 0)
        {
          printf("|");
        }
        if (sudoku[currentSudoku + i * n + j] <= 15)
        {
          printf("%X ", sudoku[currentSudoku + i * n + j]);
        }
        else
        {
          char c = 'F' + (sudoku[currentSudoku + i * n + j] - 15);
          printf("%c ", c);
        }
      }
      printf("|\n");
      if (i == n - 1)
      {
        printf("----------------------\n\n");
      }
    }
  }
}

// Print the expanded sudoku on device.
__global__ void printExpandedSudokuKernel(int *sudoku, int sqrtN, int expand)
{
  int n = sqrtN * sqrtN;
  printf("#############################\nEXPANDED SUDOKU\n#############################\n");
  for (int k = 0; k < expand; ++k)
  {
    printf(
        "#############################\nSUDOKU %d\n#############################\n", k);
    int currentSudoku = k * n * n;
    for (int i = 0; i < n; ++i)
    {
      if (i % sqrtN == 0)
      {
        printf("----------------------\n");
      }
      for (int j = 0; j < n; ++j)
      {
        if (j % sqrtN == 0)
        {
          printf("|");
        }
        if (sudoku[currentSudoku + i * n + j] <= 15)
        {
          printf("%X ", sudoku[currentSudoku + i * n + j]);
        }
        else
        {
          char c = 'F' + (sudoku[currentSudoku + i * n + j] - 15);
          printf("%c ", c);
        }
      }
      printf("|\n");
      if (i == n - 1)
      {
        printf("----------------------\n\n");
      }
    }
  }
}

void waitForErrors()
{
  do
  {
    sleep(2);
    printf("Looking for errors...\n");
    cudaError_t error = cudaGetLastError();
    printf("Error: %s\n", cudaGetErrorString(error));
    if (error != cudaSuccess)
    {
      exit(1);
    }
  } while (true);
}