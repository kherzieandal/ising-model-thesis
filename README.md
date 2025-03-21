### Overview

---

This repository contains my undergraduate thesis titled "Parallel Monte Carlo Simulation of the 2D Ising Model using GPU and CPU", submitted to the Polytechnic University of the Philippines in partial fulfillment of the requirements for the Degree Bachelor of Science in Physics.

### Abstract

---

The checkerboard Metropolis algorithm is executed utilizing numerous CPU and mobile GPU threads and cores. The specifications of the computing system include an Intel i5-11400H CPU, 16GB RAM, Nvidia RTX 3050 Laptop GPU, and Tesla T4. Python is used as the programming language using Jupyter Notebooks. The Python interpreter and Python compiler could both be outperformed by the parallel-CPU at speeds that are $390\times$ and $2.61\times$ faster, respectively. When underutilized, the parallel-CPU outperforms the GPU by up to $7.70\times$. For high lattice sizes, the mobile GPU can outperform the parallel CPU by up to $11.97\times$. The Tesla T4 outperforms the mobile GPU by up to $1.87\times$. The parallel-CPU and the GPUs need enough memory to simulate extremely high lattice sizes. The kernel crashes for parallel-CPU, and the GPUs run into memory allocation errors when running very high lattices. The average magnetization is calculated once the system has been equilibrated. To verify the correctness of our simulation, we compare it to the analytical solution provided by Onsager. Unlike previous studies, our code incorporates the interaction constant and external magnetic field as variables. Additionally, we successfully built a GPU computing rig.

### Repository Structure

---

```
├── code/                     # Code used for analysis/simulations
├── results/                  # Figures, tables, or supplementary materials
├── README.md                 # This readme file
└── LICENSE                   # License information
```

### Citation

---

```
@mastersthesis{andal2023parallel,
    author = {Kherzie Andrei Andal and Rhenish Simon},
    title = {Parallel Monte Carlo Simulation of the 2D Ising Model using GPU and CPU},
    city = {Manila},
    institution = {Polytechnic University of the Philippines},
    year = {2023},
    type = {Bachelor's Thesis}
}
```

}

### Related Publication

---

```
@inproceedings{SPP-2023-PA-06,
   author = {Kherzie Andrei G Andal and Rhenish C Simon},
   booktitle = {Proceedings of the Samahang Pisika ng Pilipinas},
   pages = {SPP-2023-PA-06},
   title = {Parallel Monte Carlo simulation of the 2D Ising model using CPU and mobile GPU},
   volume = {41},
   url = {https://proceedings.spp-online.org/article/view/SPP-2023-PA-06},
   year = {2023}
}
```

### License

---

This work is licensed under the MIT License. You are free to use, modify, and distribute it under the terms found in `LICENSE`.

### Contact

---

For any questions or discussions, feel free to contact me at <a href="mailto:kagandal@iskolarngbayan.pup.edu.ph">kagandal@iskolarngbayan.pup.edu.ph</a> or open an issue in this repository.
