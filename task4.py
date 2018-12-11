from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from mshr import *

def boundProblemSolve(alpha, dt, f, g, h, u_n, mesh, V):
	def boundary(x, on_boundary):
		return on_boundary and (x[1]<=0)
	bc = DirichletBC(V, h, boundary)

	u = TrialFunction(V)
	v = TestFunction(V)
	a = dt*dot(grad(u), grad(v))*dx+alpha*u*v*dx
	L = (dt*f+u_n)*v*dx+dt*g*v*ds

	u = Function(V)
	solve(a == L, u, bc)
	
	return u

'''
alpha = 1
R = 1
degree = 2

domain = Circle(Point(0, 0), R)
mesh = generate_mesh(domain, 100)
V = FunctionSpace(mesh, 'P', degree)


solNumber = 1
exactSol = Expression('2*exp(3*x[0])-30*x[1]*x[1]', degree = degree)
f = Expression('60+6*exp(3*x[0])-360*x[1]*x[1]', degree = degree)
g = Expression('(6*exp(3*x[0])*x[0]-60*x[1]*x[1])/R', degree = degree, R = R)
h = Expression('2*exp(3*x[0])-30*x[1]*x[1]', degree = degree)

solNumber = 2
exactSol = Expression('x[0]*x[0]+x[1]*x[1]', degree = degree)
f = Expression('x[0]*x[0]+x[1]*x[1]-4', degree = degree)
g = Expression('2*(x[0]*x[0]+x[1]*x[1])/R', degree = degree, R = R)
h = Expression('x[0]*x[0]+x[1]*x[1]', degree = degree)

solNumber = 3
exactSol = Expression('sin(x[0]+x[1])-cos(2*x[1])', degree = degree)
f = Expression('4*sin(x[0]+x[1])-6*cos(2*x[1])', degree = degree)
g = Expression('((x[0]+x[1])*cos(x[0]+x[1])+2*x[1]*sin(2*x[1]))/R', degree = degree, R = R)
h = Expression('sin(x[0]+x[1])-cos(2*x[1])', degree = degree)


u = boundProblemSolve(alpha, 1, f, g, h, 0, mesh, V)

u_e = interpolate(exactSol, V)
error_L2 = errornorm(u_e, u, 'L2')
vertex_values_exactSol = u_e.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
error_C = np.max(np.abs(vertex_values_u - vertex_values_exactSol))

print("L2-error = ", error_L2)
print("C-error = ", error_C)

# подготовка к визуализации
k1 = mesh.num_vertices()
k2 = mesh.geometry().dim()
mesh_coordinates = mesh.coordinates().reshape((k1, k2))
triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)
ax = plt.gca()
ax.set_aspect(1./ax.get_data_ratio()) # квадратная картинка


zfaces = np.asarray([u(cell.midpoint()) for cell in cells(mesh)])
plt.tripcolor(triangulation, facecolors=zfaces)
plt.savefig('u_' + str(solNumber) + '.png')

zfaces = np.asarray([u_e(cell.midpoint()) for cell in cells(mesh)])
plt.tripcolor(triangulation, facecolors=zfaces)
plt.savefig('u_e' + str(solNumber) + '.png')
'''