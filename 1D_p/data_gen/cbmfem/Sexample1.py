def faSh1(x,p):
    import numpy as np
    N=len(x)
    h=1/(N)
    goal=np.zeros((N,1))
    A=np.zeros((N,N),dtype=complex)    
    i,j = np.indices(A.shape)
    A[i==j] = 2
    A[i==j-1] = -1
    A[i==j+1] = -1
    A[i==j-i-1+3*N*i]=-2
    A=A/h
    def f(x):
        M=len(x)
        g=np.zeros((M,1),dtype=complex)
        g=((x**2)*(x**2-p))
        return g
    def F(x):       
        return A@x+h*f(x)-goal
    def df(x):
        g1=np.identity(len(x))*((4*(x**(3)))-2*p*x)
        return g1
    def dF(x):
        return A+h*df(x)
    return [F,dF]