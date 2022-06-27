import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_legendre
from scipy import integrate
import math as mt


dim=10001
sep=dim
m=dim

D=np.zeros((dim,dim))
x1=-1
dx=2/sep
x = np.linspace(-1, 1, num=sep)

for n in range(0,sep):
    w=sep-1
    for l in range(0,m):
        pol=eval_legendre(l,x1)
        fun=pol
        #m1=dx*(2*l+1)/6
        m1=dx/3
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
bt=1
N=20

gaf=np.loadtxt('202mgmcgb.txt',usecols=0,skiprows=1,delimiter=', ')
print(np.shape(gaf))
z=np.zeros((dim))
a=100

x = np.linspace(-1, 1, num=2001)
ang=np.arccos(x)
grad=ang*180/(mt.pi) #cambiamos de radianes a grados
r=2*a*np.sin(ang/2)
np.place(r, r==0, [1e-9])
k=1.0/9.6
b=1510.01 #amplitud del potencial
#b=0
al=1.0/0.1 #alfa
u=b*np.exp(-k*r)*(1-np.exp(-al*r))/(r)
cf=np.exp(gaf-bt*u)-1-gaf
ex=np.exp(gaf-bt*u)
    #print(cf)
    #calculamos las sumatorias:

c=np.zeros((m))

for l in range(0,m):
    sum=0.0
    for n in range(0,sep):
        sum+=D[l,n]*cf[n]
    c[l]=sum

print('usando la matriz obtenemos',c[10])

c2=np.zeros((m))
for l in range(0,m):
    #vamos a calcular los coeficientes cm
    pol=(eval_legendre(l, x))
    y = pol*cf
    c2[l]=(integrate.simpson(y, dx=dx,even='avg'))
print('Usando simpson obtenemos:',c2[10])

plt.plot(c,'r-*')
plt.plot(c2,'b-*')
plt.show()
