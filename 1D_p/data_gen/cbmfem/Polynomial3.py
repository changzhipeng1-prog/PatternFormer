import numpy as np
def Initial_guess3(x,p):
    import numpy as np
    from numpy.linalg import eig
    if len(x)==0:
        F=np.zeros((1,4),dtype=complex)  
        for j in range(1):
            A=np.zeros((4,4),dtype=complex)  
            a,b = np.indices(A.shape)
            A[a==b+1] = 1
            A[1,3]=2*(-(1)**2)
            A[2,3]=p*((1)**2)
            w,v=eig(A)
            F[0,4*j:4*(j+1)]=w.copy()         
    else:    
        soln=len(x[0,:])
        lamda=1
        N=2*len(x[:,0])
        F=np.zeros((N,4*soln),dtype=complex)         
        for cc in range(soln):
            F[::2,4*cc:4*(cc+1)]=x[:,cc:cc+1].copy()
        for i in range(len(x)):
            if i<len(x)-1:
                for j in range(soln):
                    A=np.zeros((4,4),dtype=complex)  
                    a,b = np.indices(A.shape)
                    A[a==b+1] = 1
                    A[0,3]=(x[i,j]+x[i+1,j])*((N)**2)/lamda
                    A[1,3]=2*(-(N)**2)
                    A[2,3]=p
                    w,v=eig(A)
                    F[2*i+1,4*j:4*(j+1)]=w.copy()
            else :
                for j in range(soln):
                    A=np.zeros((4,4),dtype=complex)  
                    a,b = np.indices(A.shape)
                    A[a==b+1] = 1
                    A[0,3]=(x[-1,j])*((N)**2)/lamda
                    A[1,3]=2*(-(N)**2)
                    A[2,3]=p
                    w,v=eig(A)
                    F[-1,4*j:4*(j+1)]=w.copy()                       
    return F