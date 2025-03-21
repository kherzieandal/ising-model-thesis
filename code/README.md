### Basic Implementation using Python

### Required packages:

-   numpy
-   numba
-   cupy
-   matplotlib (optional, for plotting only)
-   mpi4py (for multiple GPUs)

Additional info:

-   You might get an `OSerror` for CUDA IPC on windows, so I would recommend to use either WSL or Linux.

### Basic Usage

Python

```

```

Numba-compiled

```

```

Parallel-CPU

```

```

Single GPU:

```
python ising_gpu.py -x <rows> -y <columns> -n <number of iterations>
```

Multi GPU using MPI:

```
mpirun -np <# of GPUS> python ising_gpu.py -x <rows> -y <columns> -n <number of iterations>
```

Run `python ising_gpu.py --help` for more options.

### Visualizing Results

`-o` flag enables output of final lattice configuration to text files `final_rank*.txt`. Use provided `plot_ising_multi.py` to visualize output.

For example:

```
$ mpirun -np 2 python ising_gpu.py -x 2048 -y 2048 -n 100 -a 0.5 -o
...
Writing lattice to final_rank0.txt...
Writing lattice to final_rank1.txt...

$ python plot_ising_multi.py
```

This will produce the following output:

![sample_plot.png](sample_plot.png)
