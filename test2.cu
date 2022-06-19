#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

int isSolved = 0;

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

// Check if the cell is valid.
int isCellValidDevice(int *sudoku, int value, int i, int j, int sqrtN)
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
int *iterativeBruteforce(int *sudoku, int sqrtN, int *stackSpace)
{
    int n = sqrtN * sqrtN;
    int i = 0;                // current cell
    int stackIndexesSize = 0; // number of elements in stackSpace
    while (i < n * n)         // while there are still cells to fill
    {
        // if sudoku has been solved already, return
        if (isSolved == 1)
        {
            return NULL;
        }

        // get first empty cell (ignore cells with value != 0)
        if (sudoku[i] != 0)
        {
            i++;
            continue;
        }

        int foundOneValid; // if a valid value has been found for a cell
        do
        {
            foundOneValid = 0;
            int row = i / n, col = i % n; // get the row and column of the cell
            // printSudoku(sudoku, sqrtN);
            // printf("%d : %d %d\n", i, row, col);

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
                    stackSpace[stackIndexesSize++] = i;
                    break;
                }
            }
            
            // //print stackSpace
            // for (int j = 0; j < stackIndexesSize; ++j)
            // {
            //     printf("%d | ", j, stackSpace[j]);
            // }
            // printf("\n");


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

    // if is solved, return
    if (isSudokuSolved(sudoku, sqrtN) == 1)
    {
        return sudoku;
    }
    else
    {
        return NULL;
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

int *expandSudoku(int *sudoku, int sqrtN, int expand, int *count)
{
    int n = sqrtN * sqrtN;
    if (expand <= 1)
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

    // create solution array on device
    int *solutionDevice = (int *)malloc(n * n * sizeof(int));

    int *stackSpace = (int *)malloc(sizeof(int) * n * n);  // each thread has its own stack (n*n)

    int *result = iterativeBruteforce(sudoku, sqrtN, stackSpace);
    if (result != NULL)
    {
        isSolved = 1;
    }

    printf("isSolved: %d\n", isSolved);

    if (isSolved == 1)
    {
        printSudoku(result, sqrtN);
    }
    else
    {
        printf("Sudoku not solved!\n");
    }

    free(sudoku);
    return 0;
}