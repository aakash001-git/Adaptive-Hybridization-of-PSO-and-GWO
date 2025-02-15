# Adaptive-Hybridization-of-PSO-and-GWO
This Project is based on the research paper by Narinder Singh and S. B. Singh, " Hybrid Algorithm of Particle Swarm Optimization and Grey Wolf Optimizer for Improving Convergence Performance", Journal of Applied Mathematics, vol. 2017, pp. 1-15, 2017.  

The main idea is to balance the exploration and exploitation ability of Particle Swarm Optimization(PSO) and Grey Wolf Optimizer(GWO) to produce both variants’ strength by introducing the additional adaptive parameters 'lambda' which initially allows exploration with the help of PSO and later exploitation with the help of GWO with some sort of exploration also. Even though the original hybrid approach exists, it is appropriate only for problems with a continuous search space and few local optimum. However, feature selection and slow convergence speed and easy to trap into local optimum on some problems or escaping multiple local minima and narrow steep valleys is a problem and therefore, Adaptive Hybridization of PSO and GWO is proposed and implemented.  

We have used 7 standard benchmark functions to evaluate the performance of this AHPSOGWO. The results show that AHPSOGWO is significantly better than the original GWO and PSO. The AHPSOGWO improved the global optimum value in some of the functions, while in all the others, it reached the global optimum value in fewer iterations escaping multiple local minima than the original PSO and GWO.
# Components
PLOT.py file plots the graph of all three optimization algorithms' performance with " optimum value vs iterations " benchmarks.py file consist of 7 benchmark's function which can be used to evaluate/compare all three algorithms.
# How to Use:
open PLOT.py file and set the value of variable " function_to_be_used " from F1 to F7 representing the 7 unimodal and multiomodal benchmark function.  
run PLOT.py file. Use command - python PLOT.py
