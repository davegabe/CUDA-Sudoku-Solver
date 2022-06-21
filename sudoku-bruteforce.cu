#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

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
__global__ void bruteforceKernel(int *expandedSudoku, int sqrtN, int *solution, int *sudokuSpace, int *emptyCells, int nEmptyCells, int *isSolvedDevice)
{
  int n = sqrtN * sqrtN;
  int totalCells = n * n;
  int cellsPerThread = totalCells / blockDim.x;
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  // Each thread is responsible for a separate search tree.
  int startID = threadIdx.x * cellsPerThread; // Get first cell assigned to the thread.
  // printf("Executing expanded sudoku #%d cells in [%d, %d] on thread #%d\n", blockIdx.x, startID, startID + cellsPerThread - 1, threadIdx.x);

  // For each cell assigned to the thread, try all possible values.
  int *sudokuToTest = sudokuSpace + threadId * totalCells; // Get the sudoku space the thread is responsible for.

  // Copy the expanded sudoku to the sudoku space the thread is responsible for.
  for (int i = 0; i < totalCells; ++i)
  {
    sudokuToTest[i] = expandedSudoku[blockIdx.x * totalCells + i];
  }
  // Test id the sudoku is already solved.
  if (isSudokuSolved(sudokuToTest, sqrtN) == 1)
  {
    *isSolvedDevice = 1;
    memcpy(solution, sudokuToTest, totalCells * sizeof(int));
    return;
  }

  // For each cell assigned to the thread, try all possible values.
  for (int id = startID; id - startID <= cellsPerThread; ++id) // For each cell assigned to the thread.
  {
    if (sudokuToTest[id] != 0)
    {
      continue;
    }
    // move this cell to the end of the array of empty cells to remove it
    for (int i = 0; i < nEmptyCells; ++i)
    {
      if (emptyCells[threadId * nEmptyCells + i] == id)
      {
        emptyCells[threadId * nEmptyCells + i] = emptyCells[threadId * nEmptyCells + nEmptyCells - 1];
        emptyCells[threadId * nEmptyCells + nEmptyCells - 1] = id;
        break;
      }
    }
    for (int testValue = 1; testValue <= n; ++testValue) // Try all possible values.
    {
      // if sudoku has been solved already, return
      if (*isSolvedDevice == 1)
      {
        return;
      }
      int row = id / n, col = id % n; // get the row and column of the cell
      // if the cell can be filled with the value, fill it and try to solve the sudoku.
      if (isCellValidDevice(sudokuToTest, testValue, row, col, sqrtN) == 1)
      {
        sudokuToTest[id] = testValue;
        // if the cell is valid, try next cell
        int isSolved = iterativeBruteforce(sudokuToTest, sqrtN, emptyCells + threadId * nEmptyCells, nEmptyCells - 1, isSolvedDevice);
        if (isSolved == 1)
        {
          *isSolvedDevice = 1;
          memcpy(solution, sudokuToTest, totalCells * sizeof(int));
          return;
        }
        sudokuToTest[id] = 0;
      }
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
      // create sudoku "solution" and set value to cell
      int *solution = (int *)malloc(n * n * sizeof(int));
      memcpy(solution, sudoku, n * n * sizeof(int));
      solution[cell] = testValue;
      // try to get another level of sudoku
      int nBFSSolutions = 0;
      if (layers > 1)
      {
        int **BFSSolutions = BFSSudoku(solution, sqrtN, &nBFSSolutions, emptyCells, nEmptyCells - 1, layers - 1);
        if (BFSSolutions != NULL)
        {
          // append the solutions to the results
          results = (int **)realloc(results, (*count + nBFSSolutions) * sizeof(int *));
          for (int i = 0; i < nBFSSolutions; ++i)
          {
            results[*count + i] = BFSSolutions[i];
          }
          *count += nBFSSolutions;
        }
      }
      else
      {
        // append the solution to the results
        results = (int **)realloc(results, (*count + 1) * sizeof(int *));
        results[*count] = solution;
        (*count)++;
      }
    }
  }
  return results;
}

// Return concatenated sudokus (rappresenting separated trees of calculation) for the given sudoku.
int *expandSudoku(int *sudoku, int sqrtN, int expand, int *count, int *emptyCells, int *nEmptyCells)
{
  int n = sqrtN * sqrtN;
  if (expand <= 0 || *nEmptyCells == 0)
  {
    // deep copy the sudoku
    int *solution = (int *)malloc(n * n * sizeof(int));
    memcpy(solution, sudoku, n * n * sizeof(int));
    *count = 1;
    return solution;
  }
  int *expandedSudoku = (int *)malloc(sizeof(int) * n * n);

  // solve first cell
  *count = 0;
  int **solutions = BFSSudoku(sudoku, sqrtN, count, emptyCells, *nEmptyCells, expand);

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

int main(int argc, char *argv[])
{
  srand(time(NULL));
  // # HOST #
  int n = 0;
  int nEmptyCells = 0;                                              // number of empty cells
  int *emptyCells = NULL;                                           // list of empty cells
  int *sudoku = readSudoku(argv[1], &n, &emptyCells, &nEmptyCells); // read sudoku from file, get the size of the sudoku and list of empty cells
  int sqrtN = sqrt(n);
  int expand = (int)atoi(argv[2]);
  int blockSize = n * n; // the number of threads per block
  int isSolved = 0;
  int realExpand = 1;
  int *expandedSudoku = NULL;

  // # DEVICE #
  int *expandedSudokuDevice = NULL;
  int *solutionDevice = NULL;
  int *sudokuSpaceDevice = NULL;
  int *emptyCellsDevice = NULL;
  int *isSolvedDevice = NULL;

  clock_t begin = clock();

  expandedSudoku = expandSudoku(sudoku, sqrtN, expand, &realExpand, emptyCells, &nEmptyCells);                // expand the sudoku filling some empty cells
  cudaMalloc((void **)&expandedSudokuDevice, realExpand * n * n * sizeof(int));                               // alloc expanded sudoku to device
  cudaMemcpy(expandedSudokuDevice, expandedSudoku, realExpand * n * n * sizeof(int), cudaMemcpyHostToDevice); // copy expanded sudoku to device
  cudaMalloc((void **)&solutionDevice, n * n * sizeof(int));                                                  // create solution array on device
  cudaMalloc((void **)&sudokuSpaceDevice, sizeof(int) * n * n * realExpand * blockSize);                      // each thread has its own sudoku copy (n*n)
  cudaMalloc((void **)&isSolvedDevice, sizeof(int));                                                          // create isSolved on device
  cudaMemset(isSolvedDevice, 0, sizeof(int));                                                                 // set isSolved to 0

  // clone empty cells array on device
  cudaMalloc((void **)&emptyCellsDevice, sizeof(int) * nEmptyCells * realExpand * blockSize); // list of empty cells
  for (int i = 0; i < realExpand * blockSize; ++i)
  {
    cudaMemcpy(emptyCellsDevice + i * nEmptyCells, emptyCells, nEmptyCells * sizeof(int), cudaMemcpyHostToDevice);
  }
  // create the kernel and wait for it to finish
  printf("Starting kernel... (executing %d blocks with %d threads)\n", realExpand, blockSize);
  bruteforceKernel<<<realExpand, blockSize>>>(expandedSudokuDevice, sqrtN, solutionDevice, sudokuSpaceDevice, emptyCellsDevice, nEmptyCells, isSolvedDevice);
  cudaDeviceSynchronize();
  printf("Kernel finished...\n");

  // copy isSolved from device to host
  cudaMemcpy(&isSolved, isSolvedDevice, sizeof(int), cudaMemcpyDeviceToHost);

  // copy and print solution from device to host
  free(sudoku);
  cudaMemcpy(sudoku, solutionDevice, n * n * sizeof(int), cudaMemcpyDeviceToHost);

  clock_t end = clock();

  if (isSolved == 1)
  {
    printSudoku(sudoku, sqrtN);
    printf("Sudoku solved in %f seconds\n", (double)(end - begin) / CLOCKS_PER_SEC);
    cudaFree(solutionDevice);
    free(sudoku);
  }
  else
  {
    printf("Sudoku not solved!\n");
  }

  // free memory host
  free(expandedSudoku);
  free(emptyCells);
  // free memory device
  cudaFree(expandedSudokuDevice);
  cudaFree(sudokuSpaceDevice);
  cudaFree(emptyCellsDevice);

  return 0;
}