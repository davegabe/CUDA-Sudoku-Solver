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
  if (fp == NULL)
  {
    printf("Error: cannot open file %s\n", filename);
    exit(1);
  }
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
__device__ int isSudokuSolved(int *sudoku, int sqrtN)
{
  int n = sqrtN * sqrtN;
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
__device__ int isCellValidDevice(int *sudoku, int value, int i, int j, int sqrtN)
{
  int n = sqrtN * sqrtN;
  // check row
  for (int k = 0; k < n; ++k)
  {
    if (k != j && sudoku[i * n + k] == value)
    {
      return 0;
    }
  }
  // check column
  for (int k = 0; k < n; ++k)
  {
    if (k != i && sudoku[k * n + j] == value)
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
      if (k != i && l != j && sudoku[k * n + l] == value)
      {
        return 0;
      }
    }
  }
  return 1;
}

// Check if the cell is valid.
int isCellValid(int *sudoku, int value, int i, int j, int sqrtN)
{
  int n = sqrtN * sqrtN;
  for (int k = 0; k < n; ++k)
  {
    // check row
    if (k != j && sudoku[i * n + k] == value)
    {
      return 0;
    }
    // check column
    if (k != i && sudoku[k * n + j] == value)
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
      if (k != i && l != j && sudoku[k * n + l] == value)
      {
        return 0;
      }
    }
  }
  return 1;
}

// Each thread is responsible for a separate tree.
__device__ int *iterativeBruteforce(int *sudoku, int sqrtN, int *stackSpace)
{
  // printf("Thread %d\n", id);
  int n = sqrtN * sqrtN;
  int i = 0;
  while (i < n * n)
  {
    // if sudoku has been solved already, return
    if (isSolvedDevice == 1)
    {
      return NULL;
    }
    // get first empty cell
    if (sudoku[i] != 0)
    {
      i++;
      continue;
    }

    // get the row and column of the cell
    // printf("Thread %d | row: %d, col: %d | %d\n", blockIdx.x, row, col, sudoku[i]);

    int stackIndexesSize = 0;
    int foundOneValid;
    do
    {
      foundOneValid = 0;
      int row = i / n, col = i % n;
      // if the cell is empty, try all possible values
      for (int testValue = sudoku[i] + 1; testValue <= n; ++testValue)
      {
        if (isCellValidDevice(sudoku, testValue, row, col, sqrtN) == 1)
        {
          // printf("OK TestValue: %d | %d\n", testValue, i);
          // set value to cell
          sudoku[i] = testValue;
          foundOneValid = 1;

          // push the cell to the stack
          stackSpace[stackIndexesSize++] = i; //!! NON RISOLVE I SUDOKU NEMMENO QUELLI VUOTI, SI FERMA A 15 CELLE OGNI VOLTA
          break;
        }
      }

      // if no value is valid, go back to the previous modified cell
      if (foundOneValid == 0)
      {
        sudoku[i] = 0;
        if (stackIndexesSize > 0)
        {
          i = stackSpace[--stackIndexesSize];
        }
        else
        {
          i = n * n;
        }
      }
    } while (foundOneValid == 0 && i < n * n); // repeat going back until a valid value is found
    i++;
  }

  // if is solved, notfy all threads and return
  if (isSudokuSolved(sudoku, sqrtN) == 1)
  {
    return sudoku;
  }
  else
  {
    return NULL;
  }
}

// Kernel to call recursion on device.
__global__ void bruteforceKernel(int *expandedSudoku, int sqrtN, int *solution, int *sudokuSpace, int *stackSpace)
{
  int n = sqrtN * sqrtN;
  int totalCells = n * n;
  int cellsPerThread = totalCells / blockDim.x;

  // Each thread is responsible for a separate execution tree.
  int startID = threadIdx.x * cellsPerThread; // Get first cell assigned to the thread.
  // printf("Executing expanded sudoku #%d cells in [%d, %d]\n", blockIdx.x, startID, startID + cellsPerThread - 1);

  // printf("STARTID %d, CELLSPERTHREAD %d\n", startID, cellsPerThread);
  // For each cell assigned to the thread, try all possible values.
  int *sudokuToTest = sudokuSpace + (blockIdx.x * blockDim.x + threadIdx.x) * totalCells ; // Get the sudoku sèace the thread is responsible for.

  for (int id = startID; id - startID <= cellsPerThread; ++id)
  {
    // printf("[TESTING] %d: %d\n", id, sudokuToTest[id]);
    for (int testValue = 1; testValue <= n; ++testValue)
    {
      for (int i = 0; i < totalCells; ++i) // !! questo è messo qui per test, sarebbe da riportare fuori ai due cicli e farlo solo una volta
      {
        sudokuToTest[i] = expandedSudoku[blockIdx.x * blockDim.x * totalCells + i];
      }
      // printf("[TESTING] %d: %d\n", id, testValue);
      // if sudoku has been solved already, return
      if (isSolvedDevice == 1)
      {
        return;
      }
      if (sudokuToTest[id] == 0 && isCellValidDevice(sudokuToTest, testValue, id / n, id % n, sqrtN) == 1)
      {
        // printf("Ok testValue:%d, %d\n", id, testValue);
        sudokuToTest[id] = testValue;
        // if the cell is valid, try next cell
        int *result = iterativeBruteforce(sudokuToTest, sqrtN, stackSpace + (blockIdx.x * blockDim.x + threadIdx.x) * totalCells);
        // printf("RESULT: %d \n", result);
        if (result != NULL)
        {
          isSolvedDevice = 1;
          memcpy(solution, result, totalCells * sizeof(int));
          return;
        }
        sudokuToTest[id] = 0;
      }
    }
  }
}

// Get a list of possible sudoku for the given sudoku.
int **bfsSudokuSolver(int *sudoku, int sqrtN, int *count)
{
  int n = sqrtN * sqrtN;
  int **results = NULL;
  *count = 0;

  // find good values for an empty cell
  // get an empty cell
  int idx = 0;
  do
  {
    idx = rand() % (n * n);
  } while (sudoku[idx] != 0);
  // test every possible value
  for (int testValue = 1; testValue <= n; ++testValue)
  {
    if (isCellValid(sudoku, testValue, idx / n, idx % n, sqrtN) == 1)
    {
      int *solution = (int *)malloc(n * n * sizeof(int));
      memcpy(solution, sudoku, n * n * sizeof(int));
      solution[idx] = testValue;
      results = (int **)realloc(results, ((*count) + 1) * sizeof(int *));
      results[(*count)++] = solution;
    }
  }
  return results;
}

int *expandSudoku(int *sudoku, int n, int expand, int *count)
{
  if (expand <= 1)
  {
    // deep copy the sudoku
    int *solution = (int *)malloc(n * n * sizeof(int));
    memcpy(solution, sudoku, n * n * sizeof(int));
    *count = 1;
    return solution;
  }
  int sqrtN = sqrt(n);
  int *expandedSudoku = (int *)malloc(sizeof(int) * n * n);

  // solve first cell
  *count = 0;
  int **solutions = bfsSudokuSolver(sudoku, sqrtN, count);
  expand -= *count;

  // if not enough solutions, solve next cells
  for (int i = 0; i < *count && expand > 0; i++)
  {
    int *sudokuToExpand = solutions[i];
    int nextCount = 0;
    int **nextSolutions = bfsSudokuSolver(sudokuToExpand, sqrtN, &nextCount);

    // append the next solutions to solutions
    solutions = (int **)realloc(solutions, ((*count) + nextCount) * sizeof(int *));
    for (int j = 0; j < nextCount; j++)
    {
      solutions[*count + j] = nextSolutions[j];
    }
    expand -= nextCount;
    (*count) += nextCount;
  }

  // copy the solutions to the expanded sudoku
  expandedSudoku = (int *)realloc(expandedSudoku, (*count) * n * n * sizeof(int));
  for (int i = 0; i < *count; i++)
  {
    memcpy(expandedSudoku + i * n * n, solutions[i], n * n * sizeof(int));
  }
  return expandedSudoku;
}

int main(int argc, char *argv[])
{
  srand(time(NULL));
  int n = 0;
  int *sudoku = readSudoku(argv[1], &n);
  int sqrtN = sqrt(n);
  int expand = (int)atoi(argv[2]);
  int blockSize = n * n; // the number of threads per block

  int isSolved = 0;
  cudaMemcpyToSymbol(isSolvedDevice, &isSolved, sizeof(int), 0, cudaMemcpyHostToDevice);

  // expand the sudoku filling some empty cells
  int realExpand = 1;
  int *expandedSudoku = expandSudoku(sudoku, n, expand, &realExpand);
  // copy to device
  int *expandedSudokuDevice = NULL;
  cudaMalloc((void **)&expandedSudokuDevice, realExpand * n * n * sizeof(int));
  cudaMemcpy(expandedSudokuDevice, expandedSudoku, realExpand * n * n * sizeof(int), cudaMemcpyHostToDevice);
  printExpandedSudoku(expandedSudoku, n, realExpand);
  // printExpandedSudokuKernel<<<1, 1>>>(expandedSudokuDevice, sqrtN, realExpand);

  // create solution array on device
  int *solutionDevice = NULL;
  cudaMalloc((void **)&solutionDevice, n * n * sizeof(int));

  // create the kernel and wait for it to finish
  printf("Starting kernel... expanded sudoku: %d\n", realExpand);
  // cudaDeviceSetLimit(cudaLimitMallocHeapSize, sizeof(int) * n * n * blockSize);
  int *sudokuSpace = NULL;
  cudaMalloc((void **)&sudokuSpace, sizeof(int) * n * n * realExpand * blockSize); // each thread has its own sudoku copy (n*n)
  int *stackSpace = NULL;
  cudaMalloc((void **)&stackSpace, sizeof(int) * n * n * realExpand * blockSize); // each thread has its own stack (n*n)
  bruteforceKernel<<<realExpand, blockSize>>>(expandedSudokuDevice, sqrtN, solutionDevice, sudokuSpace, stackSpace);
  // waitForErrors();
  cudaDeviceSynchronize();
  printf("Kernel finished...\n");

  // copy and print isSolved from device to host
  cudaMemcpyFromSymbol(&isSolved, isSolvedDevice, sizeof(int), 0, cudaMemcpyDeviceToHost);

  printf("isSolved: %d\n", isSolved);

  // copy and print solution from device to host
  cudaMemcpy(sudoku, solutionDevice, n * n * sizeof(int), cudaMemcpyDeviceToHost);

  if (isSolved == 1)
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