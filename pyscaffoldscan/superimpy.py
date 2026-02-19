from numpy import zeros, array, eye, vstack, linalg, equal, c_, r_, dot, isnan, around 

def calc_a_P(Xc,Yc):
    """Setup variables for superposition eigenproblem"""
    ### For speed, this version unrolls the following
    ### A += cross(a,b)
    ### P += outer(a,b) + outer(b,a) - 2*dot(a,b)*identity(3)
    A  = zeros(3)
    P  = zeros([3,3])

    for k in range(len(Yc)):
        a = Yc[k]
        b = Xc[k]
        a0b0 = a[0]*b[0]
        a1b1 = a[1]*b[1]
        a2b2 = a[2]*b[2]
        a1b1 = a[1]*b[1]
        a1b2 = a[1]*b[2]
        a2b1 = a[2]*b[1]
        a2b0 = a[2]*b[0]
        a0b2 = a[0]*b[2]
        a0b1 = a[0]*b[1]
        a1b0 = a[1]*b[0]
        a0b1a1b0 = a0b1 + a1b0
        a0b2a2b0 = a0b2 + a2b0
        a1b2a2b1 = a1b2 + a2b1
        A  += array([ a1b2 - a2b1, a2b0 - a0b2, a0b1 - a1b0 ]) 
        P  += array([ [-2*a1b1 -2*a2b2, a0b1a1b0, a0b2a2b0], [a0b1a1b0, -2*a0b0 -2*a2b2, a1b2a2b1], [a0b2a2b0,  a1b2a2b1, -2*a0b0 -2*a1b1] ]) 
    return A,P

def superpose_rot_trans(X,Y):
    """Takes the N x 3 numpy array coordinates of X and Y, aligns X onto Y, and returns the rotation matrix and translation vector"""
    EYE = eye(3)

    if equal(X,Y).all():
        print('Warning: superpose_rot_trans was called with identical coordinates')
        print(X[0])
        print(Y[0])
        return [EYE, array([0.,0.,0.])]

    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    a,P = calc_a_P(Xc,Yc)

    gamma = vstack( (c_[P,a], r_[a,zeros(1)]) )
    SSD = []
    D,V = linalg.eig(gamma)

    EYE = eye(3)

    ## Detect if there is no rotation (catch upcoming divide by zero)
    if abs(V[-1]).min() < 0.00001:
        print('WARNING: superpose_rot_trans no rotation')
        return [EYE, Y.mean(0) - X.mean(0)]
        
    Vnorm = V/V[-1,:]

    bestSSD = 999999
    for k in range(4):
        S = array( [[0,           -Vnorm[2,k],  Vnorm[1,k]], 
                    [Vnorm[2,k],   0,          -Vnorm[0,k]],
                    [-Vnorm[1,k],  Vnorm[0,k],  0        ]] )

        if linalg.det(EYE-S)!=0: #invertible, non-singular
            Q = linalg.inv( dot( EYE+S, linalg.inv(EYE-S) ) )
            SSD = ((dot(Q,Xc.T).T - Yc)**2).sum()
            if isnan(SSD):
                print('Warning: no rotation just translation')
                return [EYE,[0,0,0]]
            if SSD < bestSSD:
                bestSSD = SSD
                bestS = S.copy()
        else:
            ## Problems when optimally rotated blocks try to superimpose, resulting in a non-invertable martix
            print('Warning: ignoring singularity')

    rotation = linalg.inv(dot(EYE+bestS,linalg.inv(EYE-bestS)))
    translation = Y.mean(0) - dot(rotation, X.mean(0))
    R = rotation.T
    R = around(rotation.T, decimals=12)
    T = around(translation, decimals=12)
    return R, T



