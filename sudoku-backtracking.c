#include <dirent.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SUDOKU_PATH "./sudoku-examples/"

// Read the sudoku from file.
int **readSudoku(const char *filename, int *n)
{
  FILE *fp = fopen(filename, "r");
  int **sudoku = NULL;
  char line[1024];

  // read the size of the sudoku
  if (fgets(line, sizeof(line), fp))
  {
    char *scan = line;
    sscanf(scan, "%d", n);
    sudoku = malloc(*n * sizeof(int *));

    // read the sudoku
    for (int i = 0; i < *n; ++i)
    {
      sudoku[i] = calloc(*n, sizeof(int));
      for (int j = 0; j < *n; ++j)
      {
        int t = -1;
        if (fscanf(fp, "%X", &t) != 1)
        {
          char c;
          fscanf(fp, "%c", &c);
          c = tolower(c);
          t = (c - 'f') + 15;
        }
        // value of cell is the same as the value in the file
        sudoku[i][j] = t;
      }
    }

    fclose(fp);

    return sudoku;
  }
  return NULL;
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

// Solve iteratively the sudoku.
int iterativeSolveSudoku(int **sudoku, int n)
{
  int sqrtN = sqrt(n);
  int *emptyCells = NULL;  // array of empty cells
  int countEmptyCells = 0; // number of empty cells
  int currentCell = 0;     // current cell to be filled

  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      if (sudoku[i][j] == 0)
      {
        emptyCells = realloc(emptyCells, (countEmptyCells + 1) * sizeof(int));
        emptyCells[countEmptyCells++] = i * n + j;
      }
    }
  }

  while (currentCell < countEmptyCells) // while there are still cells to fill
  {
    int id = emptyCells[currentCell];
    int foundOneValid; // if a valid value has been found for a cell
    do
    {
      foundOneValid = 0;
      int row = id / n, col = id % n; // get the row and column of the cell

      // if the cell is empty, try all possible values
      for (int testValue = sudoku[row][col] + 1; testValue <= n; ++testValue)
      {
        if (isCellValid(sudoku, testValue, row, col, n) == 1)
        {
          // set value to cell
          sudoku[row][col] = testValue;
          foundOneValid = 1;

          // go to the next cell
          currentCell++;
          break;
        }
      }

      // if no value is valid, go back to the previous modified cell
      if (foundOneValid == 0)
      {
        sudoku[row][col] = 0;
        if (currentCell > 0)
        {
          id = emptyCells[--currentCell];
        }
        else
        {
          currentCell = countEmptyCells;
        }
      }
      // repeat going back until a valid value is found
    } while (foundOneValid == 0 && currentCell < countEmptyCells);
  }

  // if is solved, return 1
  if (isSolved(sudoku, sqrtN) == 1)
  {
    return 1;
  }
  else
  {
    return 0;
  }
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
      int n = 0;
      int **sudoku = readSudoku("./sudoku-examples/sudoku.txt", &n);

      clock_t begin = clock();
      int isSolved = iterativeSolveSudoku(sudoku, n);
      clock_t end = clock();
      total_time += (end - begin);

      if (isSolved == 1)
      {
        printSudoku(sudoku, n);
        printf("Sudoku solved in %f seconds\n",
               (double)(end - begin) / CLOCKS_PER_SEC);
      }
      else
      {
        printf("No solution found\n");
      }
    }
    closedir(d);
    printf("############################################################\n");
    printf("Solved %d sudokus in %f seconds\n", n_sudokus, (double)total_time / CLOCKS_PER_SEC);
    printf("Average time: %f seconds\n", (double)total_time / CLOCKS_PER_SEC / n_sudokus);
  }
  return 0;
}