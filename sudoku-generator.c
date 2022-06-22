#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Check if the cell is valid.
int isCellValid(int **sudoku, int value, int i, int j, int n)
{
  int sqrtN = sqrt(n);
  for (int k = 0; k < n; ++k)
  {
    // check row
    if (k != j && sudoku[i][k] == value)
    {
      return 0;
    }
    // check column
    if (k != i && sudoku[k][j] == value)
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
      if (k != i && l != j && sudoku[k][l] == value)
      {
        return 0;
      }
    }
  }
  return 1;
}

// Init sudoku cells to 0.
void initSudoku(int **sudoku, int n)
{
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      sudoku[i][j] = 0;
    }
  }
}

// Complete sudoku cells.
int completeSudoku(int **sudoku, int n) //!! TO FIX
{
  for (int row = 0; row < n; ++row)
  {
    for (int val = 1; val <= n; ++val)
    {
      int col;
      do
      {
        col = rand() % n;
      } while (sudoku[row][col] != 0 || isCellValid(sudoku, val, row, col, n) == 0); // repeat if cell is not empty or value is not valid
      sudoku[row][col] = val;
    }
  }
  return 1;
}

// Generate a random sudoku.
void generateSudoku(int **sudoku, int n)
{
  // initialize sudoku to 0
  initSudoku(sudoku, n);

  // complete sudoku
  completeSudoku(sudoku, n);

  // print sudoku
  for (int k = 0; k < n; k++)
  {
    for (int j = 0; j < n; j++)
    {
      printf("%d ", sudoku[k][j]);
    }
    printf("\n");
  }
  // // remove some cells
  // int quantity = (n * n * (rand() % 40)) / 100; // max 60% of cells are removed
  // for (int i = 0; i < quantity; i++)
  // {
  //   int r, c;
  //   do
  //   {
  //     int id = rand() % (n * n);
  //     r = id / n;
  //     c = id % n;
  //   } while (sudoku[r][c] == 0);
  //   sudoku[r][c] = 0;
  // }
}

// Output the sudoku to path as a text file.
void sudokuToFile(int **sudoku, int n, char *path, int i)
{
  char *fileName = malloc(sizeof(char) * (strlen(path) + 30));
  sprintf(fileName, "%s/sudoku-%dx%d-%d.txt", path, n, n, i);
  printf("%s\n", fileName);
  FILE *file = fopen(fileName, "w");
  fprintf(file, "%d\n", n);
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      fprintf(file, "%d ", sudoku[i][j]);
    }
    fprintf(file, "\n");
  }
  fclose(file);
}

int main(int argc, char *argv[])
{
  if (argc != 4)
  {
    printf("Usage: %s <size of sudoku> <numebr of sudokus> <output path>\n", argv[0]);
    return 1;
  }
  int n = (int)atoi(argv[1]);
  int quantity = (int)atoi(argv[2]);
  int **sudoku = (int **)malloc(sizeof(int *) * n);
  for (int i = 0; i < n; i++)
  {
    sudoku[i] = (int *)malloc(sizeof(int) * n);
  }
  char *path = argv[3];
  for (int i = 0; i < quantity; i++)
  {
    printf("Generating sudoku %d/%d\n", i + 1, quantity);
    generateSudoku(sudoku, n);
    printf("Saving to file... ");
    sudokuToFile(sudoku, n, path, i);
  }
  return 0;
}