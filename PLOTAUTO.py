import matplotlib.pyplot as plt
from AHPSOGWOAUTO import NEWAHPSOGWOAUTO
from GWOAUTO import GWOAUTO
import benchmarks
from PSOAUTO import PSOAUTO

function_to_be_used = benchmarks.F4

gwo = GWOAUTO(function_to_be_used)
gwo.opt()
x1 = gwo.return_result()
print("GWO",x1)

ahpsogwo =NEWAHPSOGWOAUTO(function_to_be_used)
ahpsogwo.opt()
x2 = ahpsogwo.return_result()
print("AHPSOGWO",x2)

pso = PSOAUTO(function_to_be_used)
pso.random_init()
pso.start()
x3 = pso.return_result()
print("PSO",x3)

# PLOT
plt.figure()
plt.plot(x1)
plt.plot(x2)
plt.plot(x3)
plt.grid()
plt.legend(["GWO", "AHPSOGWO", "PSO"], loc="upper right")
plt.title("Comparision of AHPSOGWO with PSO and GWO")
plt.xlabel("Number of Iterations")
plt.ylabel("Best Score")
plt.show()
