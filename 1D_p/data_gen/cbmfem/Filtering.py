def Filtering(S2,degree,F,dF,p,method):
    import numpy as np
    from numpy import linalg as LA 
    from Polynomial3 import Initial_guess3
    from WB1 import TF5
    from Newton import Newton
    N=len(S2[:,0])
    G=Initial_guess3(S2,p)
    ll=len(S2[0,:])
    
    for soln in range(ll):
        for ss in range(degree):
            x=G[:,degree*soln+ss:degree*soln+ss+1].copy()
            S=G[:,degree*soln+ss:degree*soln+ss+1].copy()
            ##
            y=x.copy()
            e=x[0::2,0:1].copy()
            y[1::2,0:1]=0.5*(e+np.append(e[1::1,0:1],np.zeros((1,1)),axis=0))
            tol=2*LA.norm(F(y),2)
            ##
            
            for i in range(N-1):
                erase=np.array([])
                if len(S[0,:])==0:
                    break
                for j in range(len(S[0,:])):
                    x=S[:,j:j+1].copy()
                    cons=0
                    for cccc in range(degree):
                        x[2*i+3]=G[2*i+3,degree*soln+cccc:degree*soln+cccc+1].copy()
                        if LA.norm(F(x)[0:2*i+3],2)<tol:
                            cons=1
                            if cccc>0:
                                S=np.append(S,x[:,0:1],axis=1)
                            elif cccc==0 :
                                S[:,j:j+1]=x.copy()
                        elif LA.norm(F(x)[0:2*i+3],2)>=tol:
                            if cccc==0 and cons==0:
                                if len(erase)==0:
                                    erase=[j]
                                else:
                                    erase.append(j)
                if len(erase)>0:
                    S=np.delete(S, erase, 1)
            if soln+ss==0:
                S1=S.copy()
            else :
                S1=np.append(S1,S,axis=1)
        #print(len(S1[0,:]))
        
        
    S2=np.zeros((2*N,1))
    goal=np.zeros((2*N,1))
    for iii in range(len(S1[0,:])):
        x=S1[:,iii:iii+1].copy()
        for ii in range(3000):
            ##
            if method==0:
                x=TF5(x,1,p,1,5,goal)
            elif method==1:
                x=Newton(F,dF,x,10**(-9),50000)[:-1,0:1]
            elif method==2:
                x=TF5(x,1,p,0,5,goal)
            ##
            if LA.norm(F(x),2)<10**(-9):
                break
        ncon=0
        for jjjj in range(len(S2[0,:])):
            if LA.norm((x[:,0:1]-S2[:,jjjj:jjjj+1]),2)<10**(-7):
                ncon=1
        if ncon==0 and LA.norm(F(x),2)<10**(-9):
            S2=np.append(S2,x[:,0:1],axis=1)
    erase=np.array([])
    for jjjj in range(len(S2[0,:])):
        if N<4:
            if LA.norm(F(S2[:,jjjj:jjjj+1]),2)>10**(-6):
                if len(erase)==0:
                    erase=[jjjj]
                else:
                    erase.append(jjjj)
        ##            
        else :
            if LA.norm(F(S2[:,jjjj:jjjj+1]),2)>10**(-6) or abs(S2[:,jjjj:jjjj+1].imag).max()>10**(-7):
                if len(erase)==0:
                    erase=[jjjj]
                else:
                    erase.append(jjjj)  
        ##
    if len(erase)>0:
        S2=np.delete(S2, erase, 1)        
        
        
        
        
        
    return S2