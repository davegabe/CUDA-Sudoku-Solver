#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "debug.cu"

__device__ int isSolvedDevice;

// Read the sudoku from file.
int *readSudoku(const char *filename, int *n)
{
  FILE *fp = fopen(filename, "r");
  int *sudoku = NULL;
  char line[1024];

  // read the size of the sudoku
  if (fgets(line, sizeof(line), fp))
  {
    char *scan = line;
    sscanf(scan, "%d", n);
    sudoku = (int *)malloc(*n * *n * sizeof(sudoku));

    // read the sudoku
    for (int i = 0; i < *n * *n; ++i)
    {
      int t = -1;
      if (fscanf(fp, "%X", &t) != 1)
      {
        char c;
        fscanf(fp, "%c", &c);
        c = tolower(c);
        t = (c - 'f') + 15;
      }
      sudoku[i] = t;
    }

    fclose(fp);
  }

  return sudoku;
}

// Print the sudoku.
void printSudoku(int *sudoku, int n)
{
  int sqrtN = sqrt(n);
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

// Check if the sudoku is solved.
__device__ int isSudokuSolved(int *sudoku, int n)
{
  int sqrtN = sqrt((float)n);
  int i, j, k, l;
  for (i = 0; i < n; ++i)
  {
    for (j = 0; j < n; ++j)
    {
      if (sudoku[i * n + j] == 0)
      {
        return 0;
      }
      for (k = 0; k < n; ++k)
      {
        if (k != j && sudoku[i * n + j] == sudoku[i * n + k])
        {
          return 0;
        }
      }
      for (k = 0; k < n; ++k)
      {
        if (k != i && sudoku[i * n + j] == sudoku[k * n + j])
        {
          return 0;
        }
      }
      for (k = i / sqrtN * sqrtN; k < i / sqrtN * sqrtN + sqrtN; ++k)
      {
        for (l = j / sqrtN * sqrtN; l < j / sqrtN * sqrtN + sqrtN; ++l)
        {
          if (k != i && l != j && sudoku[i * n + j] == sudoku[k * n + l])
          {
            return 0;
          }
        }
      }
    }
  }
  return 1;
}

// Check if the cell is valid.
__device__ int isCellValidDevice(int *sudoku, int i, int j, int sqrtN)
{
  int n = sqrtN * sqrtN;
  // check row
  for (int k = 0; k < n; ++k)
  {
    if (k != j && sudoku[i * n + j] == sudoku[i * n + k])
    {
      return 0;
    }
  }
  // check column
  for (int k = 0; k < n; ++k)
  {
    if (k != i && sudoku[i * n + j] == sudoku[k * n + j])
    {
      return 0;
    }
  }
  // check grid
  int startI = i / sqrtN * sqrtN;
  int startJ = j / sqrtN * sqrtN;
  for (int k = startI; k < startI + sqrtN; ++k)
  {
    for (int l = startJ; l < startJ + sqrtN; ++l)
    {
      if (k != i && l != j && sudoku[i * n + j] == sudoku[k * n + l])
      {
        return 0;
      }
    }
  }
  return 1;
}

// Check if the cell is valid.
int isCellValidExanded(int *sudoku, int i, int j, int sqrtN, int expandI)
{
  int n = sqrtN * sqrtN;
  int currentSudoku =  expandI * n * n;
  for (int k = 0; k < n; ++k)
  {
    // check row
    if (k != j && sudoku[currentSudoku + i * n + j] == sudoku[currentSudoku + i * n + k])
    {
      return 0;
    }
    // check column
    if (k != i && sudoku[currentSudoku + i * n + j] == sudoku[currentSudoku + k * n + j])
    {
      return 0;
    }
  }
  // check grid
  int startI = i / sqrtN * sqrtN;
  int startJ = j / sqrtN * sqrtN;
  for (int k = startI; k < startI + sqrtN; ++k)
  {
    for (int l = startJ; l < startJ + sqrtN; ++l)
    {
      if (k != i && l != j && sudoku[currentSudoku + i * n + j] == sudoku[currentSudoku + k * n + l])
      {
        return 0;
      }
    }
  }
  return 1;
}

// Recursion on device.
__device__ int *recursionBruteforce(int *sudoku, int sqrtN, int id)
{
  // printf("Thread %d\n", id);
  int n = sqrtN * sqrtN;

  // check if is last cell
  if (id == n * n)
  {
    printf("Last cell\n");
    return NULL;
  }

  // if sudoku has been solved already, return
  if (isSolvedDevice == 1)
  {
    printf("Sudoku already solved\n");
    return NULL;
  }

  // if is solved, notfy all threads and return
  if (isSudokuSolved(sudoku, n) == 1)
  {
    printf("Sudoku solved\n");
    isSolvedDevice = 1;
    return sudoku;
  }

  // get the row and column of the cell
  int row = id / n, col = id % n;
  // printf("Thread %d | row: %d, col: %d | %d, %d\n", id, row, col, sudoku[id], sudoku[row * n + col]);
  // if the cell is empty, try all possible values
  for (; sudoku[row * n + col] != 0; id++)
  {
    row = id / n;
    col = id % n;
  }

  for (int i = 1; i <= n; ++i)
  {
    // printf("[TESTING] %d: %d\n", id, i);
    sudoku[row * n + col] = i;
    if (isCellValidDevice(sudoku, row, col, sqrtN) == 1)
    {
      // printf("Ok i:%d, %d\n", id, i);
      // if the cell is valid, try next cell
      int *solution = recursionBruteforce(sudoku, sqrtN, id + 1);
      // printf("[END] Ok i:%d j:%d, %d, %d\n", row, col, i, solution);
      if (solution != NULL)
      {
        return solution;
      }
    }
    sudoku[row * n + col] = 0;
    // printf("Not ok i:%d j:%d, %d\n", row, col, i);
  }

  return NULL;
}

// Kernel to call recursion on device.
__global__ void bruteforceKernel(int *sudoku, int sqrtN, int *solution)
{
  // Get the thread id
  int n = sqrtN * sqrtN;
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  int *result = recursionBruteforce(sudoku, sqrtN, id);
  if (result != NULL)
  {
    // deep copy the solution to the host
    for (int i = 0; i < n * n; ++i)
    {
      solution[i] = result[i];
    }
  }
}

int *expandSudoku(int *sudoku, int n, int expand)
{
  int sqrtN = sqrt(n);
  int *expandedSudoku = (int *)malloc(sizeof(int) * n * n * expand);
  srand(time(NULL));
  // expand the sudoku filling the empty cells with random numbers between 1 and n
  for (int i = 0; i < expand; ++i)
  {
    // deep copy the sudoku to the expanded sudoku
    for (int j = 0; j < n * n; ++j)
    {
      expandedSudoku[i * n * n + j] = sudoku[j];
    }

    // get the empty cell
    int idx = 0;
    do
    {
      idx = rand() % (n * n) + i * n * n;
    } while (expandedSudoku[idx] != 0);

    // try to fill the cell with a number between 1 and n
    int r = (idx - i * n * n) / n;
    int c = (idx - i * n * n) % n;
    for (int j = 1; j <= n; ++j)
    {
      expandedSudoku[idx] = j;
      if (isCellValidExanded(expandedSudoku, r, c, sqrtN, i) == 1)
      {
        break;
      }
    }
  }
  return expandedSudoku;
}

int main(int argc, char *argv[])
{
  int n = 0;
  int *sudoku = readSudoku(argv[1], &n);
  int sqrtN = sqrt(n);
  int expand = (int)atoi(argv[2]);
  int blockSize = n;

  int isSolved = 0;
  cudaMemcpyToSymbol(isSolvedDevice, &isSolved, sizeof(int), 0, cudaMemcpyHostToDevice);
  printSudoku(sudoku, n);
  // expand the sudoku filling some empty cells
  int *expandedSudoku = expandSudoku(sudoku, n, expand);
  // copy to device
  int *expandedSudokuDevice = NULL;
  cudaMalloc(&expandedSudokuDevice, n * n * expand * sizeof(int));
  cudaMemcpy(expandedSudokuDevice, expandedSudoku, expand * n * n * sizeof(int), cudaMemcpyHostToDevice);
  // printExpandedSudoku(expandedSudoku, n, expand);
  // printExpandedSudokuKernel<<<1,1>>>(expandedSudokuDevice, sqrtN, expand);

  // create solution array on device
  int *solutionDevice = NULL;
  cudaMalloc((void **)&solutionDevice, n * n * sizeof(int));

  // create the kernel and wait for it to finish
  printf("Starting kernel\n");
  bruteforceKernel<<<expand, 1>>>(expandedSudokuDevice, sqrtN, solutionDevice);
  cudaDeviceSynchronize();

  // copy and print isSolved from device to host
  cudaMemcpyFromSymbol(&isSolved, isSolvedDevice, sizeof(int), 0, cudaMemcpyDeviceToHost);

  printf("isSolved: %d\n", isSolved);

  // copy and print solution from device to host
  cudaMemcpy(sudoku, solutionDevice, n * n * sizeof(int), cudaMemcpyDeviceToHost);

  if (sudoku != NULL)
  {
    printSudoku(sudoku, n);
  }
  else
  {
    printf("Sudoku not solved!\n");
  }

  free(sudoku);
  return 0;
}