#include<stdio.h>
#include<stdlib.h>

int isSolved;

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

int isSudokuSolved(int *sudoku, int sqrtN)
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

int is CellValid1(int *sudoku, int value,  )
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

int isRowValid(int *sudoku, int value, int i,int sqrtN)
{

    int n = sqrtN * sqrtN;
    for (int j = 0; j < n; ++j) 
    {
        if (sudoku[i][j] == value) 
        {
            return 0
        }
    }

    return 1 

}

int isColValid(int *sudoku, int value, int j,int sqrtN)
{

    int n = sqrtN * sqrtN;
    for (int i = 0; i < n; ++i) 
    {
        if (sudoku[i][j] == value) 
        {
            return 0
        }
    }

    return 1 

}

int solveSudoku()

int main(void) {

    int n = 0;
    int *sudoku = readSudoku(".sudoku-examples/sudoku.txt", &n);

    return 0;

}