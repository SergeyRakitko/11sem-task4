from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from mshr import *
import task4

alpha = 1
R = 1
degree = 2
T = 100  # финальное время
num_steps = 50  # число шагов
dt = T / num_steps  # шаг по времени
domain = Circle(Point(0, 0), R)
mesh = generate_mesh(domain, 50)
V = FunctionSpace(mesh, 'P', degree)


solNumber = 1
exactSol = Expression('x[0]*x[0]*x[0]-20*x[1]*x[1]-t', degree=degree, t=0)
f = Expression('39-6*x[0]', degree = degree, t=0)
g = Expression('(3*x[0]*x[0]*x[0]-40*x[1]*x[1])/R', degree = degree, R = R, t=0)
h = Expression('x[0]*x[0]*x[0]-20*x[1]*x[1]-t', degree=degree, t=0)
'''
solNumber = 2
exactSol = Expression('5-3*exp(2*x[0])+2*x[1]-5*t', degree=degree+2, t=0)
f = Expression('-5+12*exp(2*x[0])', degree = degree, t=0)
g = Expression('(-6*x[0]*exp(2*x[0])+2*x[1])/R', degree = degree, R = R, t=0)
h = Expression('5-3*exp(2*x[0])+2*x[1]-5*t', degree=degree, t=0)

solNumber = 3
exactSol = Expression('4*x[0]-17*sin(x[1])-3*t', degree=degree, t=0)
f = Expression('-3-17*sin(x[1])', degree = degree, t=0)
g = Expression('(4*x[0]-17*cos(x[1])*x[1])/R', degree = degree, R = R, t=0)
h = Expression('4*x[0]-17*sin(x[1])-3*t', degree=degree, t=0)
'''

u_n = interpolate(h, V)

error_L2_List = []
error_C_List = []

vtkfile = File('solution/solution.pvd')
t = 0
for n in range(num_steps):
	t += dt
	
	exactSol.t = t
	h.t = t
	f.t = t
	g.t = t
	
	u = task4.boundProblemSolve(1, dt, f, g, h, u_n, mesh, V)
	
		
	u_e = interpolate(exactSol, V)
	error_L2 = errornorm(u_e, u, 'L2')
	vertex_values_exactSol = u_e.compute_vertex_values(mesh)
	vertex_values_u = u.compute_vertex_values(mesh)
	error_C = np.max(np.abs(vertex_values_u - vertex_values_exactSol))
	
	#print("L2-error = ", error_L2)
	#print("C-error = ", error_C)

	error_L2_List.append(error_L2)
	error_C_List.append(error_C)	


	vtkfile << (u, t)
	
	u_n.assign(u)

# сохранение графиков норм
x = np.linspace(dt, T + dt, num_steps)
fig, ax = plt.subplots()
ax.plot(x, error_L2_List, label='error_L2')
ax.plot(x, error_C_List, label='error_C')
legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large')
plt.xlabel('t')
plt.savefig('Norms' + str(solNumber) + '.png')