
def Newton(f,df,x,tol,Ma):
    import numpy as np
    from numpy import linalg as LA
    ct=np.ones((1,1))
    flag=0
    while True:
        if np.isinf(df(x)).max()==1:
            flag=1
            break
        if np.isnan(df(x)).max()==1:
            flag=2
            break
        dx=-np.linalg.inv(df(x))@f(x)
        x=x+dx
        ct=ct+1
        if ct>Ma:
            flag=3
            break
        if LA.norm(f(x),2)<(tol) :
            break
    if flag==0:
        return np.append(x,ct,axis=0)
    if flag==1:
        return print(isinf)
    if flag==2:
        return print(isnan)
    if abs(f(x)).max()>0.001:
        return print(f"The itteration was too long: {ct[0]}")