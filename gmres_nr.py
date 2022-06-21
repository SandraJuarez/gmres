import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import product
from warnings import warn
import matplotlib.pyplot as plt
import math as mt
from scipy.special import eval_legendre
from scipy import integrate
import pandas as pd
from scipy.sparse.linalg import gmres

a=100 #radio

sep=int(1001)
m=int(90)
x = np.linspace(-1, 1, num=sep)
ang=np.arccos(x)
grad=ang*180/(mt.pi) #cambiamos de radianes a grados
r=2*a*np.sin(ang/2)
np.place(r, r==0, [1e-9])
kp=1.0/9.6
#b=1510.01 #amplitud del potencial
b=0.0001
n=20 #número de partículas
al=1.0/0.1 #alfa
bt=1





def cm(mc,cf,x):
    cm=np.zeros(mc)
    for mc in range(0,150):
        #vamos a calcular los coeficientes cm
        pol=(eval_legendre(mc, x))
        y = pol*cf
        cm[mc]=(2*mc+1)*(integrate.simpson(y, x))/2
    return(cm)

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
        print('Iteration: {} \t x = {} \t residual = {:.4f}'.
              format(num_iter, xs, r_norm))
        num_iter += 1
        if (r_norm<tol):
            print('Se llegó a la solución')
            break
    return xs

###############################################################################
###############################################################################
#iniciamos las matrices

dim=int(2*sep+2*m)

def genA(gaf,):
    #iniciamos nuestra matriz
    A=np.zeros((dim,dim))
    x = np.linspace(-1, 1, num=sep)
    ang=np.arccos(x)
    grad=ang*180/(mt.pi) #cambiamos de radianes a grados
    r=2*a*np.sin(ang/2)
    np.place(r, r==0, [1e-9])
    #para el potencial
    kp=1.0/9.6
    #b=1510.01 #amplitud del potencial
    b=0.0001
    n=20 #número de partículas
    al=1.0/0.1 #alfa
    bt=1
    u=b*np.exp(-kp*r)*(1-np.exp(-al*r))/(r)
    cf=np.exp(gaf-bt*u)-1-gaf
    a11=1-np.exp(gaf-bt*u)
    cm=cm(m,cf,x)

    ################################################################################
    #la parte de a11
    for t in range(0,sep):
        for s in range(0,sep):
            if(t==s):
                A[s,t]=a11[t]

    #la parte de a12
    sep2=int(2*sep)
    for t in range(sep,sep2):
        for s in range(0,sep):
            if(s==t):
                A[s,t]=1
    ###############################################################################
    #la parte de a21
    for t in range(0,sep):
        for s in range(sep,sep2):
            if(s==t):
                A[s,t]=1

    #la parte de a23
    sep3=int(sep2+m)
    x1=-1
    dx=2/sep
    for t in range(sep2,sep3):
        x1=-1
        for s in range(sep,sep2):
            s2=t-sep2
            A[s,t]=-(eval_legendre(s2, x1))
            x1=x1+dx
    #################################################################################
    #la parte de a34
    sep4=int(sep2+m+m)
    for t in range(sep3,sep4):
        for s in range(sep2,sep3):
            if(s==t):
                A[s,t]=1

    #la parte de a34
    x1=-1
    for t in range(sep,sep2):
        for s in range(sep2,sep3):
            pol=eval_legendre(s-sep2,x1)
            fun=pol
            m1=-(2*(s-sep2)+1)/2
            mod=t%2
            if(mod==0)and(t!=sep)and(t!=sep2):
                A[s,t]=m1*4/3*fun
            elif(mod!=0)and(t!=sep)and(t!=sep2):
                A[s,t]=m1*2/3*fun
            elif(t==sep):
                A[s,t]=m1*fun/3
            elif(t==sep2-1):
                A[s,t]=m1*fun/3
        x1=x1+dx
    ##############################################################################
    #la parte de a43
    for t in range(sep2,sep3):
        for s in range(sep3,sep4):
            if(s==t):
                A[s,t]=1

    #la parte de a44

    for t in range(sep3,sep4):
        for s in range(sep3,sep4):
            m1=s-sep3
            if(s==t):
                A[s,t]=-2*n/(2*m1+1)*cm[m1]*1/(1-n/(2*m1+1)*cm[m1])-(n/(2*m1+1))**2*cm[m1]**2*1/(1-n/(2*m1+1)*cm[m1])**2

    return A

##### ahora construimos el vector B ##############################################
def genB(gaf):
    #iniciamos el vector
    B=np.zeros((dim))
    x = np.linspace(-1, 1, num=sep)
    ang=np.arccos(x)
    grad=ang*180/(mt.pi) #cambiamos de radianes a grados
    r=2*a*np.sin(ang/2)
    np.place(r, r==0, [1e-9])
    #para el potencial
    kp=1.0/9.6
    #b=1510.01 #amplitud del potencial
    b=0.0001
    n=20 #número de partículas
    al=1.0/0.1 #alfa
    bt=1
    u=b*np.exp(-kp*r)*(1-np.exp(-al*r))/(r)
    cf=np.exp(gaf-bt*u)-1-gaf
    a11=1-np.exp(gaf-bt*u)
    cm=cm(m,cf,x)
    gam=np.zeros((m))
    for s in range(0,m):
        gam[s]=cm[s]*(n/(2*s+1))*cm[s]*(1.0/(1-n*cm[s]/(2*s+1)))

    P=np.zeros((sep,m))
    x1=-1
    dx=2/sep
    for t in range(0,m):
        x1=-1
        for s in range(0,sep):
            P[s,t]=(eval_legendre(t, x1))
            x1=x1+dx
    pg=np.dot(P,gam)

    x1=-1

    Ps=np.zeros((m,sep))

    for t in range(0,sep):
        for s in range(0,m):
            pol=eval_legendre(s,x1)
            fun=pol
            m1=(2*s+1)/2
            mod=t%2
            if(mod==0)and(t!=0)and(t!=sep):
                Ps[s,t]=m1*2/3*fun
            elif(mod!=0)and(t!=0)and(t!=sep):
                Ps[s,t]=m1*4/3*fun
            elif(t==0):
                Ps[s,t]=m1*fun/3
            elif(t==sep-1):
                Ps[s,t]=m1*fun/3
        x1=x1+dx
    psc=np.dot(Ps,cf)
    for s in range(0,sep):#aquí va F1
        B[s]=cf[s]+gaf[s]+1-np.exp(-bt*u[s]+gaf[s])

    for s in range(sep,sep2): #aquí va F2
        B[s]=gaf[s-sep]-pg[s-sep]

    for s in range(sep2,sep3): #aquí va F3
        B[s]=cm[s-sep2]-psc[s-sep2]

    for s in range(sep3,sep4): #aquí va F4
        m1=s-sep3
        B[s]=-(gam[s-sep3]-(n/(2*m1+1))*cm[s-sep3]**2*(1.0/(1-n*cm[s-sep3]/(2*m1+1))))

np.random.seed(0)
x0=np.random.random(dim)
#x0=np.zeros(dim)
for s in range(dim):
   x0[s]=x0[s]*1e-05
'''
for s in range(0,sep):
    x0[s]=gaf[s]
for s in range(sep,sep2):
    x0[s]=cf[s-sep]
for s in range(sep2,sep3):
    x0[s]=gam[s-sep2]
for s in range(sep3,sep4):
    x0[s]=cm[s-sep3]
'''
for
    nmax=2303
    tol=1e-09
    xs = Gmres(A, B, x0,nmax,tol)
np.savetxt('solucion.txt', np.transpose([xs]))
