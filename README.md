# Jacobi-CPU-x-CUDA
A C code of Jacobi method for solving linear system.

1) Make sure you have the correct driver for your nVidia GPU: http://www.nvidia.com/download/index.aspx.

2) Download and save the four files at the same folder.

3) Open the terminal at the folder above and execute "nvcc main.cu -o main".

4) Execute "python tester1.py" and be happy.

Obs.: once you start tester1, make sure you stop/close the program in the middle of the 100-iterations step, so you can be sure that there'll be a file to read.

5) Execute "python analyze1.py" and take a look at the results.
