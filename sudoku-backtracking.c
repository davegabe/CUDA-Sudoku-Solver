#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int **readSudoku(const char *filename, int *n) {
  FILE *fp = fopen(filename, "r");
  int **sudoku = NULL;
  char line[1024];

  // read the size of the sudoku
  if (fgets(line, sizeof(line), fp)) {
    char *scan = line;
    sscanf(scan, "%d", n);
    sudoku = malloc(*n * sizeof(int *));

    // read the sudoku
    for (int i = 0; i < *n; ++i) {
      sudoku[i] = calloc(*n, sizeof(int));
      for (int j = 0; j < *n; ++j) {
        int t = -1;
        if (fscanf(fp, "%X", &t) != 1) {
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
void printSudoku(int **sudoku, int n) {
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
      if (sudoku[i][j] <= 15) {
        printf("%X ", sudoku[i][j]);
      } else {
        char c = 'F' + (sudoku[i][j] - 15);
        printf("%c ", c);
      }
    }
    printf("|\n");
    if (i == n - 1) {
      printf("----------------------\n\n");
    }
  }
}

// Check if the sudoku is solved
int isSolved(int **sudoku, int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (sudoku[i][j] == 0) {
        return 0;
      }
    }
  }
  return 1;
}

int isCellValid(int **sudoku, int value, int i, int j, int n) {
  int sqrtN = sqrt(n);
  for (int k = 0; k < n; ++k) {
    // check row
    if (k != j && sudoku[i][k] == value) {
      return 0;
    }
    // check column
    if (k != i && sudoku[k][j] == value) {
      return 0;
    }
  }
  // check grid
  int startI = i / sqrtN * sqrtN;
  int startJ = j / sqrtN * sqrtN;
  for (int k = startI; k < startI + sqrtN; ++k) {
    for (int l = startJ; l < startJ + sqrtN; ++l) {
      if (k != i && l != j && sudoku[k][l] == value) {
        return 0;
      }
    }
  }
  return 1;
}

int iterativeSolveSudoku(int **sudoku, int n) {
  int sqrtN = sqrt(n);
  int *emptyCells = NULL;   // array of empty cells
  int countEmptyCells = 0;  // number of empty cells
  int currentCell = 0;      // current cell to be filled
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (sudoku[i][j] == 0) {
        emptyCells = realloc(emptyCells, (countEmptyCells + 1) * sizeof(int));
        emptyCells[countEmptyCells++] = i * n + j;
      }
    }
  }

  while (currentCell < countEmptyCells)  // while there are still cells to fill
  {
    int i = emptyCells[currentCell];
    int foundOneValid;  // if a valid value has been found for a cell
    do {
      foundOneValid = 0;
      int row = i / n, col = i % n;  // get the row and column of the cell

      // if the cell is empty, try all possible values
      for (int testValue = sudoku[row][col] + 1; testValue <= n; ++testValue) {
        if (isCellValid(sudoku, testValue, row, col, n) == 1) {
          // set value to cell
          sudoku[row][col] = testValue;
          foundOneValid = 1;

          // go to the next cell
          currentCell++;
          break;
        }
      }

      // if no value is valid, go back to the previous modified cell
      if (foundOneValid == 0) {
        sudoku[row][col] = 0;
        if (currentCell > 0) {
          i = emptyCells[--currentCell];
        } else {
          currentCell = countEmptyCells;
        }
      }
      // repeat going back until a valid value is found
    } while (foundOneValid == 0 && currentCell < countEmptyCells);
  }

  // if is solved, return 1
  if (isSolved(sudoku, sqrtN) == 1) {
    return 1;
  } else {
    return 0;
  }
}

int recursiveSolveSudoku(int **sudoku, int n, int cell) {
  int i = cell / n;
  int j = cell % n;
  // find the next empty cell
  for (; cell < (n * n) && sudoku[i][j] != 0; ++cell) {
    i = cell / n;
    j = cell % n;
  }

  // if it's the last cell, check if the sudoku is solved
  if (cell >= n * n) {
    if (isSolved(sudoku, n) == 1) {  // if the sudoku is solved
      return 1;
    } else {
      return 0;
    }
  }

  // for each cell, try to fill it with a value
  for (int value = 1; value <= n; ++value) {  // try all possible values
    if (isCellValid(sudoku, value, i, j, n) == 1) {
      sudoku[i][j] = value;
      if (recursiveSolveSudoku(sudoku, n, cell + 1) == 1) {
        return 1;
      }
      sudoku[i][j] = 0;
    }
  }
  return 0;
}

int main(void) {
  int n = 0;
  long total_time = 0;
  int **sudoku = readSudoku("./sudoku-examples/sudoku.txt", &n);

  printSudoku(sudoku, n);
  clock_t begin = clock();
  // int isSolved = recursiveSolveSudoku(sudoku, n, 0);
  int isSolved = iterativeSolveSudoku(sudoku, n);
  clock_t end = clock();
  total_time += (end - begin);

  if (isSolved == 1) {
    printSudoku(sudoku, n);
    printf("Sudoku solved in %f seconds\n",
           (double)(end - begin) / CLOCKS_PER_SEC);
  } else {
    printf("No solution found\n");
  }

  return 0;
}