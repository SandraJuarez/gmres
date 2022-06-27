import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import product
from warnings import warn
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
import math as mt
from scipy.special import eval_legendre
from scipy import integrate
import pandas as pd
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import lgmres

a=100 #radio

sep=int(1001)
m=sep
x = np.linspace(-1, 1, num=sep)
ang=np.arccos(x)
grad=ang*180/(mt.pi) #cambiamos de radianes a grados
r=2*a*np.sin(ang/2)
np.place(r, r==0, [1e-9])
kp=1.0/9.6
b=1510.01 #amplitud del potencial
#b=1e-12
#b=0.00001
N=20 #número de partículas
al=1.0/0.1 #alfa
bt=1







def Gmres(A,B,x0,nm,tol):
    xs=x0
    Q=np.zeros((np.size(B),nm))
    H=np.zeros((nm+1,nm))
    r0=B-np.dot(A,xs).reshape(-1)
    r_norm=np.linalg.norm(r0)
    eb=np.zeros((nm+1))

    Q[:,0]=r0/r_norm #primer vector de Krylov
    beta=np.linalg.norm(r0)
    eb[0]=beta
    num_iter=0
    #iteración de arnoldi
    for k in range(1,nm):
        v = np.dot(A,Q[:,k-1]).reshape(-1)  #generar un nuevo candidato a vector de krylov
        for j in range(k):  # restar la proyección a los vectores anteriores
            H[j,k-1] = np.dot(Q[:,j].T, v)
            v = v - H[j,k-1] * Q[:,j]
        H[k,k-1] = np.linalg.norm(v,2)
        if H[k,k-1] != 0:  # Add the produced vector to the list, unless
            Q[:,k] = v/H[k,k-1]
        #else:  # If that happens, stop iterating.
        #    print('se llegó a h menor a tol')#return Q, h
        y=np.linalg.lstsq(H,eb,rcond=None)[0]
        r_norm=np.linalg.norm(np.dot(H,y)-eb)
        xs=x0+np.dot(Q,y)
        print('Iteration: {}  \t residual = {:.9f}'.
              format(num_iter, r_norm))
        num_iter += 1
        if (r_norm<tol):
            print('Se llegó a la solución')
            break
    return xs




###############################################################################
###############################################################################
#iniciamos las matrices

dim=sep


A=np.zeros((dim,dim))
B=np.zeros((dim))
print(np.shape(B))

#cosas que van en las matrices y en B
u=b*np.exp(-kp*r)*(1-np.exp(-al*r))/(r)
#vamos a calcular la matriz Dln
D=np.zeros((dim,dim))
x1=-1
dx=2/sep
for n in range(0,sep):
    w=sep-1
    for l in range(0,m):
        pol=eval_legendre(l,x1)
        fun=pol
        m1=(2*l+1)/2
        mod=n%2
        if(mod==0)and(n!=0)and(n!=w):
            D[l,n]=m1*2/3*fun
        elif(mod!=0)and(n!=0)and(n!=w):
            D[l,n]=m1*4/3*fun
        elif(n==0):
            D[l,n]=m1*fun/3
        elif(n==w):
            D[l,n]=m1*fun/3
    x1=x1+dx
#print(D)
#nuestra primera aproximación a gamma(x)
#gaf=np.zeros((sep))
#np.random.seed(0)
#gaf=np.random.random(dim)
#gaf=gaf*1e-08
gaf=np.loadtxt('20gmcgb.txt',usecols=0,skiprows=1,delimiter=', ')
#
#y con ella calculamos c(x)
cf=np.exp(gaf-bt*u)-1-gaf
ex=np.exp(gaf-bt*u)
print(cf)
#calculamos las sumatorias:
c=np.zeros((m))
cg=np.zeros((m))
gam=np.zeros((m))
for l in range(0,m):
    sum=0.0
    sum2=0.0
    sum3=0.0
    for n in range(0,sep):
        sum=sum+D[l,n]*cf[n]
        sum2=sum2+D[l,n]*(ex[n]-1)
        sum3=sum3+D[l,n]*gaf[n]
    c[l]=sum
    cg[l]=sum2
    gam[l]=sum3

#ahora escribimos la matriz Aln
for n in range(0,sep):
    for l in range(0,m):
        A[l,n]=D[l,n]-N/(2*l+1)*((D[l,n]*ex[n])*c[l]+cg[l]*D[l,n]*(ex[n]-1))
#y por último el vector B
for l in range(0,m):
    B[l]=gam[l]-N/(2*l+1)*(c[l]+gam[l])*c[l]

print(A[-1,-1])
print(B[-1])
#np.random.seed(0)
#x0=np.random.random(dim)
#x0=x0*1e-08
x0=np.loadtxt('20gmcgb.txt',usecols=0,skiprows=1,delimiter=', ')
#x0=np.zeros(dim)
#for s in range(dim):
#    if x0[s]<=0.5:
#        x0[s]=x0[s]*1e-01
#    else:
#        x0[s]=-x0[s]*1e-01

nmax=2303
tol=1e-08
#xs = Gmres(A, B, x0,nmax,tol)
#Mp=1.0/(1.0-c)
xs=lgmres(A,B,x0,atol=1e-05,maxiter=10001)
print(xs)
#xs=xs[0]
np.savetxt('solucion2.txt', xs,fmt='%s')
#np.savetxt('solucion2.txt',np.transpose([xs]))
