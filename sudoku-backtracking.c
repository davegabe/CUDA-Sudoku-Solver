#include <stdio.h>
#include <stdlib.h>

int **readSudoku(const char *filename, int *n)
{

  FILE *fp = fopen(filename, "r");
  if (fp == NULL)
  {
    printf("Error: cannot open file %s\n", filename);
    exit(1);
  }

  int **sudoku = NULL;
  char line[1024];

  // read the size of the sudoku
  if (fgets(line, sizeof(line), fp))
  {
    char *scan = line;
    sscanf(scan, "%d", n);
    sudoku = malloc(*n * sizeof(sudoku));

    // read the sudoku
    for (int i = 0; i < *n; ++i)
    {
      sudoku[i] = calloc(*n, sizeof(int));
      for (int j = 0; j < *n; ++j)
      {
        int t;
        fscanf(fp, "%d", &t);
        // value of cell is the same as the value in the file
        (sudoku[i][j]) = t;
      }
    }

    fclose(fp);
  }

  return sudoku;
}

// Print the sudoku.
void printSudoku(int **sudoku, int n)
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
      if (sudoku[i][j] <= 15)
      {
        printf("%X ", sudoku[i][j]);
      }
      else
      {
        char c = 'F' + (sudoku[i][j] - 15);
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

int iterativeSolveSudoku(int **sudoku, int sqrtN)
{
  int n = sqrtN * sqrtN;

  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      if (sudoku[i][j] == 0)
      {
        for (int value = 0; value <= n; ++value)
        {
          if (isCellValid(sudoku, value, i, j, sqrtN) && isRowValid(sudoku, value, i, sqrtN) && isColValid(sudoku, value, j, sqrtN))
          {
            sudoku[i][j] = value;
            break;
          }
        }
      }
    }
  }

  if (isSolved(sudoku, n) == 1)
  {
    printSudoku(sudoku, n);
  }

  else
  {
    printf("No solution found");
  }
}

int recursiveSolveSudoku(int **sudoku, int sqrtN)
{
  int n = sqrtN * sqrtN;

  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      if (sudoku[i][j] == 0)
      {
        for (int value = 0; value <= n; ++value)
        {
          if (isCellValid(sudoku, value, i, j, sqrtN) && isRowValid(sudoku, value, i, sqrtN) && isColValid(sudoku, value, j, sqrtN))
          {
            sudoku[i][j] = value;
            recursiveSolveSudoku(sudoku, sqrtN);
          }
        }
      }
    }
  }

  if (isSolved(sudoku, n) == 1)
  {
    printSudoku(sudoku, n);
  }

  else
  {
    printf("No solution found");
  }
}

// Check if the sudoku is solved
int isSolved(int **sudoku, int n)
{
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      if (sudoku[i][j] == 0)
      {
        return 0;
      }
    }
  }
  return 1;
}

int isCellValid(int **sudoku, int value, int i, int j, int sqrtN)
{
  for (int row = 0; row < sqrtN; ++row)
  {
    for (int col = 0; col < sqrtN; ++col)
    {
      if (sudoku[row + i][col + j] == value)
      {
        return 0;
      }
    }
  }
  return 1;
}

int isRowValid(int **sudoku, int value, int i, int sqrtN)
{

  int n = sqrtN * sqrtN;
  for (int j = 0; j < n; ++j)
  {
    if (sudoku[i][j] == value)
    {
      return 0;
    }
  }

  return 1;
}

int isColValid(int **sudoku, int value, int j, int sqrtN)
{

  int n = sqrtN * sqrtN;
  for (int i = 0; i < n; ++i)
  {
    if (sudoku[i][j] == value)
    {
      return 0;
    }
  }

  return 1;
}

int main(void)
{

  int n = 0;
  int **sudoku = readSudoku(".sudoku-examples/sudoku.txt", &n);
   iterativeSolveSudoku(**sudoku, n);
   recursiveSolveSudoku(**sudoku, n);

  return 0;
}