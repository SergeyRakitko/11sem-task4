from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from mshr import *

alpha = 9.3
domain = Circle(Point(0, 0), 1)
mesh = generate_mesh(domain, 100)
V = FunctionSpace(mesh, 'P', 1)

u_D = Expression('2 - 12*x[0]*x[1] - 25*x[1]*x[1]*x[0]', degree=3)

def boundary(x, on_boundary):
    return on_boundary# and (x[1]<=0)
bc = DirichletBC(V, u_D, boundary)

u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v))*dx+alpha*u*v*dx

f = Expression('4*sin(x[1])-8*x[0]', degree = 1)
g = Expression('11*(x[0])+9*x[1]', degree = 1)
L = f*v*dx+g*v*ds

u = Function(V)
solve(a == L, u, bc)

error_L2 = errornorm(u_D, u, 'L2')
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
error_C = np.max(np.abs(vertex_values_u - vertex_values_u_D))

print("L2-error = ", error_L2)
print("C-error = ", error_C)

plot(u)
plt.savefig('u4.png')
plt.show()

