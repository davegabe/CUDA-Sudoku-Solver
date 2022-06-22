#include <dirent.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "debug.cu"

#define EXPAND_PERCENTAGE 60
#define SUDOKU_PATH "./sudoku-examples/"

// Read the sudoku from file.
int *readSudoku(const char *filename, int *n, int **emptyCells, int *nEmptyCells)
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
      if (sudoku[i] == 0)
      {
        (*nEmptyCells)++;
        (*emptyCells) = (int *)realloc(*emptyCells, (*nEmptyCells) * sizeof(int));
        (*emptyCells)[(*nEmptyCells) - 1] = i;
      }
    }

    fclose(fp);
  }

  return sudoku;
}

// Print the sudoku.
void printSudoku(int *sudoku, int sqrtN)
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

// Check if the sudoku is solved.
__device__ int isSudokuSolved(int *sudoku, int sqrtN)
{
  int n = sqrtN * sqrtN;
  for (int i = 0; i < n * n; ++i)
  {
    if (sudoku[i] == 0)
    {
      return 0;
    }
  }
  return 1;
}

// Check if the cell is valid.
__device__ int isCellValidDevice(int *sudoku, int value, int i, int j, int sqrtN)
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
__device__ int iterativeBruteforce(int *sudoku, int sqrtN, int *emptyCells, int nEmptyCells, int *isSolvedDevice)
{
  int n = sqrtN * sqrtN;
  int currentCell = 0;              // current cell watching in the stack
  while (currentCell < nEmptyCells) // while there are still cells to fill
  {
    int i = emptyCells[currentCell];
    int foundOneValid; // if a valid value has been found for a cell
    do
    {
      // if sudoku has been solved already, return
      foundOneValid = 0;
      int row = i / n, col = i % n; // get the row and column of the cell

      // if the cell is empty, try all possible values
      for (int testValue = sudoku[i] + 1; testValue <= n; ++testValue)
      {
        if (*isSolvedDevice == 1)
        {
          return NULL;
        }
        if (isCellValidDevice(sudoku, testValue, row, col, sqrtN) == 1)
        {
          // set value to cell
          sudoku[i] = testValue;
          foundOneValid = 1;

          // go to the next cell
          currentCell++;
          break;
        }
      }

      // if no value is valid, go back to the previous modified cell
      if (foundOneValid == 0)
      {
        sudoku[i] = 0;
        if (currentCell > 0)
        {
          i = emptyCells[--currentCell];
        }
        else
        {
          currentCell = nEmptyCells;
        }
      }
    } while (foundOneValid == 0 && currentCell < nEmptyCells); // repeat going back until a valid value is found
  }

  // if is solved, return
  if (isSudokuSolved(sudoku, sqrtN) == 1)
  {
    return 1;
  }
  else
  {
    return 0;
  }
}

// Kernel to call recursion on device.
__global__ void bruteforceKernel(int sqrtN, int *solution, int *sudokuSpace, int *emptyCells, int nEmptyCells, int *isSolvedDevice)
{
  // Each thread is responsible for a separate search tree.
  int n = sqrtN * sqrtN;
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  int *myEmptyCells = emptyCells + threadId * nEmptyCells; // Get the empty cells for this thread.
  int *mySudoku = sudokuSpace + threadId * n * n;          // Get the sudoku space for this thread.

  // Test if the sudoku is already solved.
  if (isSudokuSolved(mySudoku, sqrtN) == 1)
  {
    *isSolvedDevice = 1;
    memcpy(solution, mySudoku, n * n * sizeof(int));
    return;
  }

  // Each thread has assigned an empty cell
  // Move it to the end of the array and start iterating without it
  int emptyCell = myEmptyCells[threadIdx.x];
  myEmptyCells[threadIdx.x] = myEmptyCells[nEmptyCells - 1];
  myEmptyCells[nEmptyCells - 1] = emptyCell;
  int row = emptyCell / n, col = emptyCell % n; // get the row and column of the cell

  for (int testValue = 1; testValue <= n; ++testValue) // Try all possible values.
  {
    // if sudoku has been solved already, return
    if (*isSolvedDevice == 1)
    {
      return;
    }
    // if the cell can be filled with the value, fill it and try to solve the sudoku.
    if (isCellValidDevice(mySudoku, testValue, row, col, sqrtN) == 1)
    {
      mySudoku[emptyCell] = testValue;
      // if the cell is valid, try next cell
      int isSolved = iterativeBruteforce(mySudoku, sqrtN, myEmptyCells, nEmptyCells - 1, isSolvedDevice);
      if (isSolved == 1)
      {
        *isSolvedDevice = 1;
        memcpy(solution, mySudoku, n * n * sizeof(int));
        return;
      }
      mySudoku[emptyCell] = 0;
    }
  }
}

// Get layers of BFS search from given cell of sudoku.
int **BFSSudoku(int *sudoku, int sqrtN, int *count, int *emptyCells, int nEmptyCells, int layers)
{
  if (nEmptyCells == 0)
  {
    return NULL;
  }

  int n = sqrtN * sqrtN;
  int **results = NULL;
  int cell = emptyCells[nEmptyCells - 1];

  // test every possible value
  for (int testValue = 1; testValue <= n; ++testValue)
  {
    if (isCellValid(sudoku, testValue, cell / n, cell % n, sqrtN) == 1)
    {
      sudoku[cell] = testValue;
      // try to get another level of sudoku
      int nBFSSolutions = 0;
      if (layers > 1)
      {
        int **BFSSolutions = BFSSudoku(sudoku, sqrtN, &nBFSSolutions, emptyCells, nEmptyCells - 1, layers - 1);
        if (BFSSolutions != NULL)
        {
          // append the solutions to the results
          results = (int **)realloc(results, (*count + nBFSSolutions) * sizeof(int *));
          memcpy(results + *count, BFSSolutions, nBFSSolutions * sizeof(int *));
          *count += nBFSSolutions;
        }
      }
      else
      {
        // append the sudoku (a solution) to the results
        results = (int **)realloc(results, (*count + 1) * sizeof(int *));
        results[*count] = (int *)malloc(n * n * sizeof(int));
        memcpy(results[*count], sudoku, n * n * sizeof(int));
        (*count)++;
      }
      sudoku[cell] = 0;
    }
  }
  return results;
}

// Return concatenated sudokus (rappresenting separated trees of calculation) for the given sudoku.
int *expandSudoku(int *sudoku, int sqrtN, int expandCells, int *count, int *emptyCells, int nEmptyCells)
{
  int n = sqrtN * sqrtN;
  if (expandCells <= 0 || nEmptyCells == 0)
  {
    // deep copy the sudoku
    int *solution = (int *)malloc(n * n * sizeof(int));
    memcpy(solution, sudoku, n * n * sizeof(int));
    *count = 1;
    return solution;
  }
  int *expandedSudoku = (int *)malloc(sizeof(int) * n * n);

  *count = 0;

  // for each empty cell, try to get a new sudoku
  for (int i = 0; i < expandCells; ++i)
  {
    for (int j = 0; j < expandCells; ++j)
    {
      if (i == j)
      {
        continue;
      }
      for (int testValue = 1; testValue <= n; ++testValue)
      {
        int cell = emptyCells[nEmptyCells - 1];
        int row = cell / n, col = cell % n;
        // if the cell can be filled with the value, fill it and try to get a new sudoku
        if (isCellValid(sudoku, testValue, row, col, sqrtN) == 1)
        {
          sudoku[cell] = testValue;
        }
      }
    }
  }

  int **solutions = BFSSudoku(sudoku, sqrtN, count, emptyCells, nEmptyCells, expandCells);

  // copy the solutions to the expanded sudoku
  expandedSudoku = (int *)realloc(expandedSudoku, (*count) * n * n * sizeof(int));
  for (int i = 0; i < *count; i++)
  {
    memcpy(expandedSudoku + i * n * n, solutions[i], n * n * sizeof(int));
    free(solutions[i]);
  }
  free(solutions);
  return expandedSudoku;
}

int main()
{
  DIR *d;
  struct dirent *dir;
  const char *path = SUDOKU_PATH;

  d = opendir(path);
  long total_time = 0;
  int n_sudokus = 0;
  if (d)
  {
    while ((dir = readdir(d)) != NULL)
    {
      if (dir->d_name[0] == '.')
        continue;

      n_sudokus++;
      char fileSudoku[100];
      sprintf(fileSudoku, "%s%s", path, dir->d_name);
      printf("%s\n", fileSudoku);

      // # HOST #
      int n = 0;
      int nEmptyCells = 0;                                                 // number of empty cells
      int *emptyCells = NULL;                                              // list of empty cells
      int *sudoku = readSudoku(fileSudoku, &n, &emptyCells, &nEmptyCells); // read sudoku from file, get the size of the sudoku and list of empty cells
      int sqrtN = sqrt(n);
      int expandCells = nEmptyCells * EXPAND_PERCENTAGE / 100; // expand EXPAND_PERCENTAGE% of empty cells
      int isSolved = 0;
      int realExpand = 1;
      int *expandedSudoku = NULL;

      // # DEVICE #
      int *expandedSudokuDevice = NULL; // expanded sudoku
      int *solutionDevice = NULL;       // address of the solution (ref sudokuSpace)
      int *sudokuSpaceDevice = NULL;    // sudoku space for each thread
      int *emptyCellsDevice = NULL;     // empty cells on sudoku
      int *isSolvedDevice = NULL;       // is sudoku solved (int)

      clock_t begin = clock();
      expandedSudoku = expandSudoku(sudoku, sqrtN, expandCells, &realExpand, emptyCells, nEmptyCells); // expand the sudoku filling some empty cells
      nEmptyCells -= expandCells;
      cudaMalloc((void **)&expandedSudokuDevice, realExpand * n * n * sizeof(int));                               // alloc expanded sudoku to device
      cudaMemcpy(expandedSudokuDevice, expandedSudoku, realExpand * n * n * sizeof(int), cudaMemcpyHostToDevice); // copy expanded sudoku to device
      cudaMalloc((void **)&solutionDevice, n * n * sizeof(int));                                                  // create solution array on device
      cudaMalloc((void **)&isSolvedDevice, sizeof(int));                                                          // create isSolved on device
      cudaMemset(isSolvedDevice, 0, sizeof(int));                                                                 // set isSolved to 0

      // clone expanded sudoku array on device
      cudaMalloc((void **)&sudokuSpaceDevice, sizeof(int) * n * n * realExpand * nEmptyCells); // each thread has its own sudoku copy (n*n)
      for (int i = 0; i < realExpand; ++i)                                                     // for each expanded sudoku (for each grid)
      {
        int *blockSudoku = expandedSudokuDevice + i * n * n; // get the grid sudoku (source to copy)
        for (int j = 0; j < nEmptyCells; ++j)                // for each thread
        {
          // i := block id | nEmptyCells := blockDim | j := thread id
          int *threadSudoku = sudokuSpaceDevice + (i * nEmptyCells + j) * n * n; // thread sudoku (destination)
          cudaMemcpy(threadSudoku, blockSudoku, n * n * sizeof(int), cudaMemcpyDeviceToDevice);
        }
      }

      // clone empty cells array on device
      cudaMalloc((void **)&emptyCellsDevice, sizeof(int) * nEmptyCells * realExpand * nEmptyCells); // list of empty cells
      for (int i = 0; i < realExpand * nEmptyCells; ++i)
      {
        cudaMemcpy(emptyCellsDevice + i * nEmptyCells, emptyCells, nEmptyCells * sizeof(int), cudaMemcpyHostToDevice);
      }
      // create the kernel and wait for it to finish
      printf("Starting kernel... (executing %d blocks with %d threads)\n", realExpand, nEmptyCells);
      bruteforceKernel<<<realExpand, nEmptyCells>>>(sqrtN, solutionDevice, sudokuSpaceDevice, emptyCellsDevice, nEmptyCells, isSolvedDevice);
      cudaDeviceSynchronize();
      printf("Kernel finished...\n");

      // copy isSolved from device to host
      cudaMemcpy(&isSolved, isSolvedDevice, sizeof(int), cudaMemcpyDeviceToHost);

      clock_t end = clock();
      total_time += (end - begin);

      if (isSolved == 1)
      {
        // copy and print solution from device to host
        cudaMemcpy(sudoku, solutionDevice, n * n * sizeof(int), cudaMemcpyDeviceToHost);
        printSudoku(sudoku, sqrtN);
        printf("Sudoku solved in %f seconds\n", (double)(end - begin) / CLOCKS_PER_SEC);
      }
      else
      {
        printf("Sudoku not solved!\n");
      }

      // free memory host
      free(expandedSudoku);
      free(emptyCells);
      free(sudoku);
      // free memory device
      cudaFree(solutionDevice);
      cudaFree(expandedSudokuDevice);
      cudaFree(sudokuSpaceDevice);
      cudaFree(emptyCellsDevice);
    }
    closedir(d);
  }
  printf("############################################################\n");
  printf("Solved %d sudokus in %f seconds\n", n_sudokus, (double)total_time / CLOCKS_PER_SEC);
  printf("Average time: %f seconds\n", (double)total_time / CLOCKS_PER_SEC / n_sudokus);

  return 0;
}