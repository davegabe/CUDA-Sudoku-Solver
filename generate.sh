rm -f ./sudoku-generator.out
gcc sudoku-generator.c -o sudoku-generator.out -lm
./sudoku-generator.out 9 100 "./sudoku-generated"