
def rotation_of_coordinate(coordinate,theta):
    """
    coordinate is either tuple or list of two numbers.
    theta in degrees.
    return ---> rotated coordinate.

    e.g.

    m = rotation_of_coordinate((8,0),45)
    print(m)
    return [6, -6]
    """
    import numpy as np
    a = np.array(coordinate)

    theta = np.deg2rad(theta)
    R = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    ap = np.matmul(R,a)
    ap = [round(l) for l in ap]
    return ap

def CIRCULAR_ARC(image , theta1,theta2,alpha,beta,r,show_plot = False):
    """
    Create a circular arc of radius r centered at (alpha,beta) between angle theta1 and theta2(>theta1)
    return 1d array of pixel values along the arc.
    """
    import numpy as np

    theta1_rad = np.deg2rad(theta1)
    theta2_rad = np.deg2rad(theta2)

    d_theta = 1/r

    n = (theta2-theta1)/d_theta     ### since rotation_of_coordinate accepts theta in degrees
    n = int(n)

    X0_p = np.array([r*np.cos(theta2_rad),r*np.sin(theta2_rad)])
    X0_p = np.int64(X0_p)
    X2_p = X0_p.copy()

    Xp = [X0_p]
    for i in range(n):
        X1_p = rotation_of_coordinate(X0_p,i*d_theta)
        if np.all(X1_p == X2_p):
            pass
        else:
            Xp.append(X1_p)
            X2_p = X1_p

    Xp = np.array(Xp)

    x = Xp.T[0]+alpha
    y = Xp.T[1]+beta

    X = np.array([x,y]).T

    V = []
    for px in X:
        V.append(image[px[1]][px[0]])

    V = np.array(V)
    # print(V[0:3],V[-3:])


    if show_plot == True:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot()
        im = ax.imshow(image,origin='lower')
        im1 = plt.plot(x,y,'-',color = 'red')
        if __name__=="__main__":
            plt.show()

    return V



if __name__=="__main__":
    import numpy as np
    img = np.random.randint(0,50,(200,150))
    theta1 = 0 ## in degrees
    theta2 = 180    ## in degrees
    alpha = 75
    beta = 0
    r = 40

    v = CIRCULAR_ARC(img,theta1=theta1,theta2=theta2,alpha=alpha,beta=beta,r=r,show_plot=True)
    
