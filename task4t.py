from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from mshr import *

alpha = 22
T = 300.0  # финальное время
num_steps = 100  # число шагов
dt = T / num_steps  # шаг по времени

domain = Circle(Point(0, 0), 1)
mesh = generate_mesh(domain, 100)
V = FunctionSpace(mesh, 'P', 1)

u_D = Expression('1 + (cos(exp(x[0])-20*x[1]))*t',
 degree=2, t=0)

def boundary(x, on_boundary):
    return on_boundary and (x[1]<=0)
bc = DirichletBC(V, u_D,boundary)
u_n = interpolate(u_D, V)

u = TrialFunction(V)
v = TestFunction(V)
a = dt*dot(grad(u), grad(v))*dx+alpha*u*v*dx
f = Expression('20*x[1]+8*x[0]-t', degree = 1, t=0)
g = Expression('9*(x[1])-11*(x[0])-sin(t)', degree = 1, t=0)
L = (dt*f+u_n)*v*dx+g*v*ds

vtkfile = File('sol/solution.pvd')

u = Function(V)
t = 0
for n in range(num_steps):
	t += dt
	u_D.t = t
	f.t = t
	g.t = t

	solve(a == L, u, bc)
	vtkfile << (u, t)
	plot(u)
	
	u_n.assign(u)