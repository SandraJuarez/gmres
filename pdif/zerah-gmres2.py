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

def Potencial(b,kp,al,a):
    x = np.linspace(-1, 1, num=sep)
    ang=np.arccos(x)
    grad=ang*180/(mt.pi) #cambiamos de radianes a grados
    r=2*a*np.sin(ang/2)
    np.place(r, r==0, [1e-9])
    #para el potencial
    u=b*np.exp(-kp*r)*(1-np.exp(-al*r))/(r)
    return u
#vamos a generar la A precondicionada
def GenA(D,N,ex,c,gam,dim,sep,m):
    A=np.zeros((dim,dim))
    for n in range(0,sep):
        for l in range(0,m):
            A[l,n]=1/(1-(N/(2*l+1))*c[l])*((1-(N/(2*l+1))*c[l])-ex[n]*(N/(2*l+1))*(gam[l]-2*c[l]))
    return A

def GenB(gam,N,c,dim,m):
    B=np.zeros((dim))
    for l in range(0,m):
        B[l]=-(gam[l]-N/(2*l+1)*(c[l]+gam[l])*c[l])
    return B

sep=int(20001)
dim=sep
m=sep

#nuestra primera aproximación a gamma(x)
#gaf=np.zeros((sep))
#np.random.seed(0)
#gaf=np.random.random(dim)
#gaf=gaf*1e-03
gaf=np.loadtxt('220120gmcgb.txt',usecols=0,skiprows=1,delimiter=', ')
#np.random.seed(0)
#x0=np.random.random(dim)
#x0=np.loadtxt('20gmcgb.txt',usecols=0,skiprows=1,delimiter=', ')
x0=np.zeros(dim)
#for s in range(dim):
#    if x0[s]<=0.5:
#        x0[s]=x0[s]*1e-12
#    else:
#        x0[s]=-x0[s]*1e-12
#x0=gaf1*1e-6

a=100 #radio
x = np.linspace(-1, 1, num=sep)
ang=np.arccos(x)
grad=ang*180/(mt.pi) #cambiamos de radianes a grados
r=2*a*np.sin(ang/2)
np.place(r, r==0, [1e-9])
kp=1.0/9.6
b=1510.01 #amplitud del potencial
#b=1e-12
#b=0.0001
N=20 #número de partículas
al=1.0/0.1 #alfa
bt=1

nmax=2303
tol=1e-9
u=Potencial(b,kp,al,a)

#vamos a generar una sola vez los polinomios de Legendre que necesitamos y guardarlos en una matriz
pol=np.zeros((dim,dim))
x1=-1
dx=2/sep
for n in range(0,sep):
    for l in range(0,m):
        pol[l,n]=eval_legendre(l,x1)
    x1=x1+dx
print('se terminó de calcular la matriz de polinomios')
#vamos a calcular la matriz Dln
D=np.zeros((dim,dim))
x1=-1
dx=2/sep
for n in range(0,sep):
    w=sep-1
    for l in range(0,m):
        fun=pol[l,n]
        m1=dx*(2*l+1)/6
        #m1=dx/3
        mod=n%2
        if(mod==0)and(n!=0)and(n!=w):
            D[l,n]=m1*2*fun
        elif(mod!=0)and(n!=0)and(n!=w):
            D[l,n]=m1*4*fun
        elif(n==0):
            D[l,n]=m1*fun
        elif(n==w):
            D[l,n]=m1*fun
    x1=x1+dx
#####################################################################################
nf=120.02
ni=120
au=0.01
fin=int((nf-ni)/au)

si=2.20
sf=2.70
aus=0.05
finsc=int((sf-si)/aus)
s=si

for sc in range(0,finsc+1):
    for nc in range(0,fin+1):
        print(N)
        for p in range(0,700):
            anterior=gaf
            cfh=np.exp(gaf-bt*u)-1-gaf
            cfp=np.exp(-bt*u)*(gaf+1)-1-gaf
            cf=s*cfh+(1-s)*cfp
            ex=np.exp(gaf-bt*u)-1
            #print(cf)
            #calculamos las sumatorias:

            c=np.zeros((m))
            gam=np.zeros((m))

            for l in range(0,m):
                sum=0.0
                for n in range(0,sep):
                    sum+=D[l,n]*cf[n]
                c[l]=sum
                gam[l]=c[l]*(N/(2*l+1))*c[l]*(1.0/(1.0-N*c[l]/(2*l+1)))


            A=GenA(D,N,ex,c,gam,dim,sep,m)
            B=GenB(gam,N,c,dim,m)
            xs = Gmres(A, B, x0,nmax,tol)

            gam=gam+xs
            sumg=0.0
            for n in range(0,sep):
                poli=pol[n,:]
                sumg=sumg+poli*gam[n]
            gaf=sumg
            name=round(s*100)
            n0=round(N*100)
            t0=str(name)+str(n0)
            if(np.max(np.abs(B))<8e-17):
                print('se llegó a la solución FINAL')
                g=gaf+cf+1
                pu=np.log(s*np.exp(gaf)+(1-s)*(1+gaf))-gaf
                np.savetxt(t0+'gmcgb.txt', np.transpose([gaf,cf,g,pu]),delimiter=',   ',header=t0)
                break
            else:
                x0=xs
        N=N+au
    s=s+aus
np.savetxt('solucion2.txt',np.transpose([gaf]))
plt.plot(grad,gaf,'g-*')
plt.plot(grad,gaf1,'r-*')
plt.show()
plt.clf()
#construimos la función de correlación
cf=np.exp(gaf-bt*u)-1-gaf
g=gaf+cf+1
np.savetxt('solucion.txt', np.transpose([gaf,cf,g]))
plt.plot(grad,g)
plt.show()
