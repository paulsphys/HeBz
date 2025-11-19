import numpy as np
from scipy.special import gamma, kv
from scipy.spatial.distance import cdist
from importlib import resources

def benzene():
    """
    Defines the orientation of the benzene molecule
    """
    z_val = 0
    coord = []
    benzene = f"""
    C    -1.2073830   -0.6970829    {z_val} 
    C    -1.2073830    0.6970829    {z_val} 
    C     0.0000000    1.3941659    {z_val} 
    C     1.2073830    0.6970829    {z_val} 
    C     1.2073830   -0.6970829    {z_val} 
    C     0.0000000   -1.3941659    {z_val} 
    H    -2.1490090   -1.2407309    {z_val} 
    H    -2.1490090    1.2407309    {z_val} 
    H     0.0000000    2.4814619    {z_val} 
    H     2.1490090    1.2407309    {z_val} 
    H     2.1490090   -1.2407309    {z_val} 
    H     0.0000000   -2.4814619    {z_val}"""
    for line in benzene.split("\n"):
        if line:
            atom, x, y, z = line.split()
            if atom == 'C':
                coord.append([0,float(x),float(y),float(z)])
            elif atom == 'H':
                coord.append([1,float(x),float(y),float(z)])
    return coord

def generate_benzene_geometry():
    """
    Generate coordinates for benzene molecule: 6 carbon + 6 hydrogen atoms for Shirkov 2024 (different orientation than ours)
    """
    rc = 1.3915  # Carbon ring radius (in Angstrom)
    rh = 1.080   # C–H bond length (in Angstrom)
    rh1 = rc + rh

    # Constants used in hexagonal arrangement
    ax = [1, 0.5, -0.5, -1, -0.5, 0.5]
    ay = [0, np.sqrt(3)/2, np.sqrt(3)/2, 0, -np.sqrt(3)/2, -np.sqrt(3)/2]

    coords = np.zeros((3, 12))  # (x, y, z) for 12 atoms

    # Carbon atoms
    for i in range(6):
        coords[0, i] = ax[i] * rc
        coords[1, i] = ay[i] * rc

    # Hydrogen atoms
    for i in range(6):
        coords[0, i+6] = ax[i] * rh1
        coords[1, i+6] = ay[i] * rh1

    return coords


def LennardJones(x,y,z):
    """
    Returns the sum of pairwise interactions between He and all the Benzene atoms in K.
    """
    coord = benzene()
    LJ = np.array([[2.98, 18.36],[2.70, 12.13]])
    val = 0
    for atom in coord:
        r2 = (x-atom[1])**2 + (y-atom[2])**2 + (z - atom[3])**2
        eps = LJ[atom[0]][1]
        sig = LJ[atom[0]][0]
        val += 4*eps*(sig**12 / (r2)**6 - sig**6 / (r2**3))

    return val

def Lee2003(x,y,z):
    """
    Returns the fitted potential derived in Lee et. al., . J. Chem. Phys. 22 December 2003; 119 (24): 12956–12964. https://doi.org/10.1063/1.1628217
    """
    def w(r):
        r0 = 7.220527
        a = 0.572446
        return 1 - np.exp(-a*(r - r0))
    def wtild(r):
        r0 = 7.220527
        if isinstance(r, (list, tuple, np.ndarray)):
            res = np.zeros(len(r))
            for i in range(len(res)):
                if (r[i] >= r0):
                    res[i] = w(r[i])
                else:
                    res[i] = 0
            return res
        else:
            if r >= r0:
                return w(r)
            else:
                return 0
    
    def V2(r):
        c3 = 0.058387
        c4 = -6.914851
        c5 = -2.083808
        c6 = 85.429568
        fval = w(r)*w(r) + c6*(wtild(r)**6)
        fval += c3*(w(r)**3) + c4*(w(r)**4) + c5*(w(r)**5)
        return fval
    def V3(rk,rl):
        fval = c11*w(rk)*w(rl) + c22*(w(rk)**2)*(w(rl)**2)
        fval += c12*(w(rk)*(w(rl)**2) + w(rl)*(w(rk)**2))
        fval += c13*(w(rk)*(w(rl)**3) + w(rl)*(w(rk)**3))
        fval += c14*(w(rk)*(w(rl)**4) + w(rl)*(w(rk)**4))
        fval += c23*((w(rk)**2)*(w(rl)**3) + ((w(rl)**2)*(w(rk)**3)))
        return fval 
    def V4(rk,rl,rm):
        fval = c111*w(rk)*w(rl)*w(rm)
        fval += c122*(w(rk)*(w(rl)**2)*(w(rm)**2) + w(rl)*(w(rm)**2)*(w(rk)**2) + w(rm)*(w(rk)**2)*(w(rl)**2))
        fval += c112*(w(rk)*w(rl)*(w(rm)**2) + w(rk)*(w(rl)**2)*w(rm) + (w(rk)**2)*w(rl)*w(rm)) 
        fval += c113*(w(rk)*w(rl)*(w(rm)**3) + w(rk)*(w(rl)**3)*w(rm) + (w(rk)**3)*w(rl)*w(rm))
        return fval
    c3 = 0.058387
    c4 = -6.914851
    c5 = -2.083808
    c6 = 85.429568
    c11 = -16.956997
    c12 = 0.568429
    c22 = 9.073184
    c13 = 1.821920
    c14 = 1.874776
    c23 = -0.098396
    c111 = -3.235122
    c112 = -2.631504
    c122 = 1.423777
    c113 = -2.117152
    C0 = 6*(1 + c3 + c4 + c5 + c6) + 15*(c11 + c22 + 2*(c12+c13+c14+c23)) + 20*(c111++3*(c112+c122+c113))
    val = 0
    coord = benzene()
    coord = coord[:6]
    for i in range(len(coord)):
        W0 = 0.023767
        bz = 1.264890
        atom = coord[i]
        rk = np.sqrt((x-atom[1])**2 + (y-atom[2])**2 + bz*(z - atom[3])**2)
        val += W0*V2(rk)
        for j in range(i):
            atoml = coord[j]
            rl = np.sqrt((x-atoml[1])**2 + (y-atoml[2])**2 + bz*(z - atoml[3])**2)
            val += W0*V3(rk,rl)
            for m in range(j):
                atomm = coord[m]
                rm = np.sqrt((x-atomm[1])**2 + (y-atomm[2])**2 + bz*(z - atomm[3])**2)
                val += W0*V4(rk,rl,rm)

    return val/0.695 #Conversion to K

def read_parameters(filename="params.txt"):
    """Read PES parameters from a file and convert to atomic units."""
    with resources.open_text("HeBz", filename) as f:
        lines = f.readlines()
    xpar = []
    for line in lines:
        p = line.split()
        xpar.append(float(p[1]))
    return xpar

def Shirkov2024(x,y,z):
    par = read_parameters()
    x0 = generate_benzene_geometry()
    # Rotate x,y,z by 30 degrees to maintain consistency with the other potentials.
    xr = x * np.cos(-30.0 * np.pi/180.0) - y * np.sin(-30.0 * np.pi/180.0)
    yr = x * np.sin(-30.0 * np.pi/180.0) + y * np.cos(-30.0 * np.pi/180.0)
    x = xr
    y = yr
    # Pre-allocate outputs
    na = 6
    na2 = 12

    # Cartesian to Spherical conversion
    r = np.sqrt(x**2 + y**2 + z**2)
    ph0 = np.arctan2(y,x)
    th0 = np.arccos(z/r)
    tt = np.cos(th0)
    fi = ph0
    #print(r,ph0,th0,tt,fi)
    # Extract parameters
    (re, ae, c0, v0, c3, c4, c5, c6, c7, c8, c12, c112, c1122, c1112, c11112,
     c11122, c111222, c111112, c111122, c123, c1123, c11223, c11123, c112233,
     reh, aeh, v0h, d3, d4, d5, d6, d12, d112, d1122, d1112, dc12, dc112,
     r0, gama, de, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12) = par[:52]

    are = ae / re
    areh = aeh / reh

    # Initialize interaction terms
    v_terms = np.zeros(15)
    v123_terms = np.zeros(5)
    vdw = 0.0

    # First triple loop for atoms 1 to na
    for k in range(na):
        dxk = x - x0[0, k]
        dyk = y - x0[1, k]
        dzk = z - x0[2, k]
        rrk = dxk**2 + dyk**2 + dzk**2
        rk = np.sqrt(rrk)
        rkk = rk - re
        rk2 = rrk / re**2
        rk6 = rk2**3
        vk = 1.0 - np.exp(-rkk * are)
        wk = vk**2 * (1 + vk * (c3 + vk * (c4 + vk * (c5 + vk * (c6 + vk * (c7 + vk * c8)))))
        )
        v_terms[0] += wk  # v11

        for l in range(k+1, na):
            dxl = x - x0[0, l]
            dyl = y - x0[1, l]
            dzl = z - x0[2, l]
            rrl = dxl**2 + dyl**2 + dzl**2
            rl = np.sqrt(rrl)
            rll = rl - re
            vl = 1.0 - np.exp(-rll * are)

            v_terms[1] += vk * vl               # v12
            v_terms[2] += vk**2 * vl + vk * vl**2     # v112
            v_terms[3] += vk**2 * vl**2               # v1122
            v_terms[4] += vk**3 * vl + vl**3 * vk     # v1112
            v_terms[5] += vk**4 * vl + vl**4 * vk     # v11112
            v_terms[6] += vk**3 * vl**2 + vl**3 * vk**2  # v11122
            v_terms[7] += vk**5 * vl + vl**5 * vk     # v111112
            v_terms[8] += vk**4 * vl**2 + vl**4 * vk**2  # v111122
            v_terms[9] += vk**3 * vl**3               # v111222

            for m in range(l+1, na):
                dxm = x - x0[0, m]
                dym = y - x0[1, m]
                dzm = z - x0[2, m]
                rm = np.sqrt(dxm**2 + dym**2 + dzm**2)
                rmm = rm - re
                vm = 1.0 - np.exp(-rmm * are)

                v123_terms[0] += vk * vl * vm
                v123_terms[1] += vk**2 * vl * vm + vk * vl**2 * vm + vk * vl * vm**2
                v123_terms[2] += vk**2 * vl**2 * vm + vk**2 * vl * vm**2 + vk * vl**2 * vm**2
                v123_terms[3] += vk**3 * vl * vm + vk * vl**3 * vm + vk * vl * vm**3
                v123_terms[4] += vk**2 * vl**2 * vm**2

    # First potential sum
    pt1 = (v_terms[0] + v_terms[1]*c12 + v_terms[2]*c112 + v_terms[3]*c1122 +
           v_terms[4]*c1112 + v_terms[5]*c11112)
    pt2 = (v_terms[6]*c11122 + v_terms[7]*c111112 + v_terms[8]*c111122 +
           v_terms[9]*c111222)
    p = pt1 + pt2 + v123_terms[0]*c123 + v123_terms[1]*c1123 + v123_terms[2]*c11223 \
        + v123_terms[3]*c11123 + v123_terms[4]*c112233

    # Reset some terms for second part
    v_terms[:5] = 0
    w_terms = np.zeros(3)  # v12, w112, w12

    # Loop for atoms 7 to na2
    for k in range(6, na2):  # 0-indexed in Python
        dxk = x - x0[0, k]
        dyk = y - x0[1, k]
        dzk = z - x0[2, k]
        rk = np.sqrt(dxk**2 + dyk**2 + dzk**2)
        rkk = rk - reh
        vk = 1.0 - np.exp(-rkk * areh)
        wk = vk**2 * (1 + vk * (d3 + vk * (d4 + vk * (d5 + vk * d6))))
        v_terms[0] += wk  # v11

        for l in range(k+1, na2):
            dxl = x - x0[0, l]
            dyl = y - x0[1, l]
            dzl = z - x0[2, l]
            rl = np.sqrt(dxl**2 + dyl**2 + dzl**2)
            rll = rl - reh
            vl = 1.0 - np.exp(-rll * areh)

            v_terms[1] += vk * vl         # v12
            v_terms[2] += vk**2 * vl + vk * vl**2  # v112
            v_terms[3] += vk**2 * vl**2           # v1122
            v_terms[4] += vk**3 * vl + vl**3 * vk  # v1112

        for l in range(na):
            dxl = x - x0[0, l]
            dyl = y - x0[1, l]
            dzl = z - x0[2, l]
            rl = np.sqrt(dxl**2 + dyl**2 + dzl**2)
            rll = rl - re
            vl = 1.0 - np.exp(-rll * are)
            w_terms[1] += vk * vl  # w12
            w_terms[2] += vk**2 * vl + vk * vl**2  # w112

    # Second potential sum
    ph = (v_terms[0] + v_terms[1]*d12 + v_terms[2]*d112 + 
          v_terms[3]*d1122 + v_terms[4]*d1112 + 
          w_terms[1]*dc12 + w_terms[2]*dc112)

    # Switching functions
    r1 = np.sqrt(x**2 + y**2 + z**2)
    h_short = 1.0 / (1.0 + np.exp(gama * (r1 - r0))) / r1
    h_long = 1.0 / (1.0 + np.exp(-gama * (r1 - r0)))

    # Angular dependence
    normm1 = np.sqrt(5.0) / 5.0
    normm2 = np.sqrt(13.0) / 13.0
    normm3 = 0.1792151994e-4

    p1 = -x1 / r**6 - x2 / r**8 - x3 / r**10 - x4 / r**12
    p2 = -normm1 * (1.5 * tt**2 - 0.5) * (x5 / r**6 + x6 / r**8 + x7 / r**10 + x8 / r**12)
    p3 = -normm2 * (3.0/8.0 + 35.0/8.0 * tt**4 - 15.0/4.0 * tt**2)
    p4 = (x9 / r**8 + x10 / r**10 + x11 / r**12)
    p5 = -normm3 * 10395.0 * (1 - tt**2)**3 * np.cos(6 * fi) * x12 / r**10
    vdw = p1 + p2 + p3 * p4 + p5

    v3 = (v0h * ph + v0 * p + c0) * h_short + vdw * h_long
    return v3/0.695

def matern_kernel(x1, x2, lengthscale):
    """
    Compute Matern kernel between x1 and x2 with given lengthscale and smoothness nu.
    """
    # pairwise distance
    r = cdist(x1 / lengthscale, x2 / lengthscale, metric='euclidean')
    #print(r)
    sqrt5 = np.sqrt(5.0)
    k = (1 + sqrt5 * r + 5 * r**2 / 3) * np.exp(-sqrt5 * r)
    return k
def kernel(X1,X2,theta1,theta2,P,scale):
    # split x and z
    #print(X1,X2)
    x1, z1 = X1[..., :-1], X1[..., -1:]
    x2, z2 = X2[..., :-1], X2[..., -1:]
    x11_ = z1
    x21t_ = z2
    x21t_ = np.transpose(x21t_)
    cross_term_1 = (1 - x11_) * (1 - x21t_)
    bias_factor = cross_term_1 * np.power(1 + x11_ * x21t_,P)
    return scale*(matern_kernel(x1,x2,lengthscale=theta1) + bias_factor*matern_kernel(x1,x2,lengthscale=theta2))

def longrange(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2)
    ph0 = np.arctan2(y,x)
    th0 = np.arccos(z/r)
    tt = np.cos(th0)
    fi = ph0
    

    x1 = 0.2290626821E+06
    x2 = 0.4069217828E+07
    x3 = 0.9542607077E+08
    x4 = 0.0000000000E+00
    x5 = -0.4775615288E+05
    x6 = -0.1263899296E+08
    x7 = -0.2959330054E+09
    x8 = 0.0000000000E+00
    x9 = 0.9511488779E+06
    x10 = 0.2551023972E+09
    x11 = 0.0000000000E+00
    x12 = 0.1883887454E+09
    

    # Angular dependence
    normm1 = np.sqrt(5.0) / 5.0
    normm2 = np.sqrt(13.0) / 13.0
    normm3 = 0.1792151994e-4

    p1 = -x1 / r**6 - x2 / r**8 - x3 / r**10 - x4 / r**12
    p2 = -normm1 * (1.5 * tt**2 - 0.5) * (x5 / r**6 + x6 / r**8 + x7 / r**10 + x8 / r**12)
    p3 = -normm2 * (3.0/8.0 + 35.0/8.0 * tt**4 - 15.0/4.0 * tt**2)
    p4 = (x9 / r**8 + x10 / r**10 + x11 / r**12)
    p5 = -normm3 * 10395.0 * (1 - tt**2)**3 * np.cos(6 * fi) * x12 / r**10
    vdw = p1 + p2 + p3 * p4 + p5
    
    return vdw

#The potential function
def V(x,y,z):
    
    #with open('data.pkl', 'rb') as file:
    #    gp_data = pickle.load(file)
    with resources.path("HeBz", "data.npz") as data_path:    
        gp_data = np.load(data_path)
    hard_wall_height = 100000;
    r0 = 0.5765366674E+01
    gama = 0.1671477473E+02
    Lx = Ly = 10;
    Lz = 20;
    Wallcz = Lz/2.0 - 1.4;
    invWallWidth = 20.0;
    vwall = 1/(1.0+np.exp(-invWallWidth*(z-Wallcz)));
    val = 0
    if (z < 0): z = -z    
    #Determine the angle the point makes with the positive x-axis
    coord = np.array([x,y,z])
    angle = np.round(np.rad2deg(np.arctan2(y,x)),4) 
    if(angle < 0): angle = 180 + (180 + angle)
    if(angle <= 30):
        xeval = x; 
        yeval = y; 
    else:
        #Determine the rotated point
        rotangle = (angle % 60) - angle
        def rotmatz(angle):
            mat = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
            return mat
        def refmatz(angle):
            mat = np.array([[np.cos(2*angle),np.sin(2*angle),0],[np.sin(2*angle),-np.cos(2*angle),0],[0,0,1]])
            return mat
        if (angle + rotangle) > 30:
            rotated = np.dot(rotmatz(np.deg2rad(rotangle)),coord)
            #print("rotated",rotated)
            final = np.dot(refmatz(np.deg2rad(30)),rotated)
            #print(final)
        else:
            final = np.dot(rotmatz(np.deg2rad(rotangle)),coord)
        #print(final)
        #Determine the potential by lookup
        xeval = final[0]; 
        yeval = final[1];
        
    coord = np.array([xeval,yeval,z])
    r = np.sqrt(xeval**2 + yeval**2)   
    r1 = np.sqrt(xeval**2 + yeval**2 + z**2) 
    h_long = 1.0 / (1.0 + np.exp(-gama * (r1 - r0)))
    #Read the file into array P
    #Evaluate_the_gp
    data_eval = np.array([[xeval,yeval,z,1]])
    #print(data_eval)
    tx = gp_data["data"]
    new_nrm = (data_eval - gp_data['xoffset'])/gp_data['xscale']
    #Predict value
    k = np.transpose(kernel(tx,new_nrm,gp_data['theta1'],gp_data['theta2'],gp_data['power'],gp_data['outputscale']))
    evalz = gp_data['mean'] + np.dot(k,gp_data['product'])
    val_gp = (gp_data['means'] + gp_data['stddevy']*evalz).flatten()
    #Join the long-range piece 
    if (r > 5 or z > 6.5):
        val += (1-h_long)*val_gp + h_long*longrange(xeval,yeval,z)
    else:
        val += val_gp
    return val[0]/0.695
