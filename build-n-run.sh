rm -f ./sudoku-bruteforce.out
nvcc sudoku-bruteforce.cu -o sudoku-bruteforce.out -lm -arch=native -g -G
./sudoku-bruteforce.out ./sudoku-examples/sudoku.txt 1