#멀티그리드 윗 function defined 
#Multigrid when it is non linear
def TF3(fa,x,k,it,goal):
    import numpy as np
    from numpy import linalg as LA
    N=len(x)
    f, df=fa(x)
    def F(x):       
        return f(x)-goal
    def dF(x):
        return df(x)
    #while abs(F(x)).max()>10**(-9):
    for repeat in range(it):
        if LA.norm(F(x),2).max()<10**(-9):
            break
        elif k==0:
            dx=-np.linalg.inv(dF(x))@F(x)
            x=x+dx
        elif k==1:
            q=(np.transpose(dF(x))@F(x)).copy()
            v=dF(x)@q
            eta=(np.transpose(v)@F(x))/(np.transpose(v)@v)
            x=x-(eta)*q[:,0:1]
        elif k==2:
            dx=-np.linalg.inv(np.diag(np.diag(dF(x))))@F(x)
            x=x+dx
        elif k==3:
            dx=-np.linalg.inv(np.tril(A))@F(x)
            x=x+dx

    #if k==3:
    if N>1: #and abs(F(x)).max()>10**(-9):
        y=x[1::2,0:1]/2+x[2::2,0:1]/4+x[:-2:2,0:1]/4
        #y=x[1::2,0:1]
        fn, dfn=fa(y)
        f1=F(x)
        f2=-1/2*f1[:-2:2,0:1]-1/2*f1[2::2,0:1]-f1[1::2,0:1]+fn(y)
        #f2=-1/4*f1[:-2:2,0:1]-1/4*f1[2::2,0:1]-1/2*f1[1::2,0:1]+fn(y)
        #f2=-f1[1::2,0:1]+fn(y)
        y1=TF3(fa,y,k,it,f2)
        e=y1-y
        x[1::2,0:1]=x[1::2,0:1]+e
        x[0::2,0:1]=x[0::2,0:1]+0.5*(np.append(np.zeros((1,1)),e,axis=0)+np.append(e,np.zeros((1,1))*e[-1,0:1],axis=0))
    #print(abs(F(x)).max())
    
    
    
    
    
#    for repeat in range(it):
#        if LA.norm(F(x),2).max()<10**(-9):
#            break
#        elif k==0:
#            dx=-np.linalg.inv(dF(x))@F(x)
#            x=x+dx
#        elif k==1:
#            q=(np.transpose(dF(x))@F(x)).copy()
#            v=dF(x)@q
#            eta=(np.transpose(v)@F(x))/(np.transpose(v)@v)
#            x=x-(eta)*q[:,0:1]
#        elif k==2:
#            dx=-np.linalg.inv(np.diag(np.diag(dF(x))))@F(x)
#            x=x+dx
#        elif k==3:
#            dx=-np.linalg.inv(np.tril(A))@F(x)
#            x=x+dx    
    
    
    
    
    return x