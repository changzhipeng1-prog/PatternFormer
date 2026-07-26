def TF5(x,lamda,p,k,it,goal):
    import numpy as np
    from numpy import linalg as LA 
    N=len(x)
    h=1/(N)
    A=np.zeros((N,N),dtype=complex)    
    #A=np.zeros((N,N))
    i,j = np.indices(A.shape)
    A[i==j] = 2
    A[i==j-1] = -1
    A[i==j+1] = -1
    A[i==j-i-1+3*N*i]=-2
    A=A/h
    def f(x):
        #M=len(x)
        #g=np.zeros((M,1),dtype=complex)
        g=((x**2)*(x**2-p))
        return g
    def F(x):       
        return A@x+h*f(x)-goal
    def df(x):
        #g1=np.zeros((N,N),dtype=complex)
        g1=np.identity(len(x))*((4*(x**(3)))-2*p*x)
        return g1
    def dF(x):
        return A+h*df(x)
    #while LA.norm(F(x),2)>10**(-9):
    for repeat in range(it):
        if LA.norm(F(x),2)<10**(-9):
            break
        elif k==0:
            dx=-np.linalg.inv(dF(x))@F(x)
            x=x+dx
        elif k==1:
            q=(np.transpose(dF(x))@F(x)).copy()
            v=dF(x)@q
            eta=(np.transpose(v)@F(x))/(np.transpose(v)@v)
            x=x-(eta)*q[:,0:1]

    #if k==3:
    if N>1: #and LA.norm(F(x),2)>10**(-9):
        y=x[0::2,0:1]
        M=len(y)
        h2=1/(M)
        #B=np.zeros((M,M),dtype=complex)    
        B=np.zeros((M,M))
        i1,j1 = np.indices(B.shape)
        B[i1==j1] = 2
        B[i1==j1-1] = -1
        B[i1==j1+1] = -1
        B[i1==j1-i1-1+3*M*i1]=-2
        B=B/h2 
        f1=F(x)
        f2=-1/2*np.append(np.zeros((1,1)),f1[1:-2:2,0:1],axis=0)-1/2*f1[1::2,0:1]-f1[0::2,0:1]+B@y+h2*f(y)
        y1=TF5(y,lamda,p,k,it,f2)
        e=y1-y
        x[0::2,0:1]=x[0::2,0:1]+e
        x[1::2,0:1]=x[1::2,0:1]+0.5*(e+np.append(e[1::1,0:1],np.zeros((1,1)),axis=0))
    #print(abs(F(x)).max())
    #print(LA.norm(F(x),2))
    return x