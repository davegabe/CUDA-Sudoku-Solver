# CUDA Sudoku Solver

## Description
Sudoku solver that uses backtracking to solve the puzzle implemented in CUDA.

## Working with hard sudoku puzzles
The operating system’s GUI watchdog timer stops the kernel after about 2 seconds of execution. There are no guarantees that the kernel will finish executing the puzzle. To solve this problem you can change the timeout value(methods are os dependent), in the next section we will see how to do this on Windows.

### Modifying TDR settings (Windows)

To modify the system TDR settings, you will need to edit the following registry key (or create it if it doesn't already exist):

```
KeyPath   : HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers
KeyValue  : TdrDelay
ValueType : REG_DWORD
ValueData : Number of seconds to delay. The default value is 2 seconds.
```

You should pick a value that is sufficiently large for your training workload to run successfully on your particular GPU.

Changes to this registry key require a system reboot to take effect.

For more information about TDR registry keys, see [Testing and debugging TDR](https://docs.microsoft.com/windows-hardware/drivers/display/tdr-registry-keys).