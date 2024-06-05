#--------------------------------------------------------------------
# Test 2D Orszag-Tang vortex problem with the MUSCL-Hancock scheme
#--------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time 

####  程式有問題，無法解決negative pressure  ####

# -------------------------------------------------------------------
# define initial condition (Orszag-Tang vortex)
# -------------------------------------------------------------------
'''
def InitialCondition(x, y):
    d = 1.0  # density

    # Velocity components
    u = -np.sin(2 * np.pi * y)
    v = np.sin(2 * np.pi * x)
    w = 0.0

    # Magnetic field components
    bx = -np.sin(2 * np.pi * y)
    by = np.sin(4 * np.pi * x)
    bz = 0.0

    # gas pressure
    P = 1.0 / gamma  

    # Total energy density
    E = P / (gamma - 1.0) + 0.5 * d * (u**2.0 + v**2 + w**2) + 0.5 * (bx**2 + by**2 + bz**2)

    # Conserved variables [0/1/2/3/4] <--> [density/momentum x/momentum y/momentum z/energy/B-field x/B-field y/B-field z/psi]
    return np.array([d, d*u, d*v, d*w, E, bx, by, bz, 0.0])
'''

# -------------------------------------------------------------------
# define initial condition (Alfan wave):跑得出東西，但感覺還是有問題
# -------------------------------------------------------------------
def InitialCondition(x, y):
    d = 1.0  # density

    # Velocity components
    u = 0.1 * np.sin(2 * np.pi * y)
    v = 0.1 * np.sin(2 * np.pi * x)
    w = 0.0

    # Magnetic field components
    bx = 1.0
    by = 0.1 * np.sin(2 * np.pi * x)
    bz = 0.0

    # gas pressure
    P = 1.0  

    # Total energy density
    E = P / (gamma - 1.0) + 0.5 * d * (u**2.0 + v**2 + w**2) + 0.5 * (bx**2 + by**2 + bz**2)

    # Conserved variables [0/1/2/3/4] <--> [density/momentum x/momentum y/momentum z/energy/B-field x/B-field y/B-field z/psi]
    return np.array([d, d*u, d*v, d*w, E, bx, by, bz, 0.0])
# -------------------------------------------------------------------
# define boundary condition by setting ghost zones
# -------------------------------------------------------------------
def BoundaryCondition(U):
    # Periodic boundary conditions
    U[:nghost, :] = U[-2*nghost:-nghost, :]
    U[-nghost:, :] = U[nghost:2*nghost, :]    
    
    U[:, :nghost] = U[:, -2*nghost:-nghost]
    U[:, -nghost:] = U[:, nghost:2*nghost]


# -------------------------------------------------------------------
# compute pressure
# -------------------------------------------------------------------
def ComputePressure(d, px, py, pz, e, bx, by, bz, psi):
    B = (bx**2.0 + by**2.0 + bz**2.0)**0.5
    P = (gamma-1.0)*( e - 0.5*(px**2.0 + py**2.0 + pz**2.0)/d - 0.5*B**2.0 ) 
    assert np.all(P > 0), "negative pressure !!"
    return P

def ComputeTotalPressure( d, px, py, pz, e, bx, by, bz, psi ):
    B = (bx**2.0 + by**2.0 + bz**2.0)**0.5
    P = (gamma-1.0)*( e - 0.5*(px**2.0 + py**2.0 + pz**2.0)/d - 0.5*B**2.0 ) + 0.5*B**2.0
    assert np.all( P > 0 ), "negative pressure !!"
    return P

# -------------------------------------------------------------------
# compute time-step by the CFL condition
# -------------------------------------------------------------------
def ComputeTimestep(U, dx, dy, cfl, gamma, end_time, t):
    P = ComputePressure(U[:,:,0], U[:, :, 1], U[:, :, 2], U[:, :, 3], U[:, :, 4], U[:, :, 5], U[:, :, 6], U[:, :, 7], U[:, :, 8])
    u = np.abs(U[:, :, 1] / U[:, :, 0])
    v = np.abs(U[:, :, 2] / U[:, :, 0])
    w = np.abs(U[:, :, 3] / U[:, :, 0])

    a = (gamma * P / U[:, :, 0])**0.5
    b = ((U[:, :, 5]**2.0 + U[:, :, 6]**2.0 + U[:, :, 7]**2.0) / U[:, :, 0])**0.5
    bx = (U[:, :, 5]**2.0 / U[:, :, 0])**0.5

    cf = ((a**2.0 + b**2.0 + ((a**2.0 + b**2.0)**2.0 - 4 * (a**2.0) * (bx**2.0))**0.5) / 2.0)**0.5

    # Maximum information speed in 3D
    max_info_speed = np.amax(u + cf)
    dt_cfl = cfl * min(dx, dy) / max_info_speed
    dt_end = end_time - t

    return min(dt_cfl, dt_end)

# -------------------------------------------------------------------
# compute limited slope
# -------------------------------------------------------------------
def ComputeLimitedSlope( L, C, R ):
#  compute the left and right slopes
   slope_L = C - L
   slope_R = R - C
   slope_LR  = slope_L*slope_R

#  apply the van-Leer limiter
#   slope_limited = np.where( slope_LR>0.0, 2.0*slope_LR/(slope_L+slope_R), 0.0 )

#  apply the MinMod limiter  
   limited_slope = np.where(slope_LR <= 0, 0, np.where(slope_L > 0, np.minimum(slope_L, slope_R), np.maximum(slope_L, slope_R)))

   return limited_slope

# -------------------------------------------------------------------
# convert conserved variables to primitive variables
# -------------------------------------------------------------------
def Conserved2Primitive( U ):
    W = np.empty( 9 )

    W[0] = U[0]         #density
    W[1] = U[1]/U[0]    # u (vx)
    W[2] = U[2]/U[0]    # v 
    W[3] = U[3]/U[0]    # w 
    W[4] = ComputePressure( U[0], U[1], U[2], U[3], U[4], U[5], U[6], U[7], U[8] )
    W[5] = U[5]         # Bx
    W[6] = U[6]         # By
    W[7] = U[7]         # Bz
    W[8] = U[8]         # psi
    
    return W

# -------------------------------------------------------------------
# convert primitive variables to conserved variables
# -------------------------------------------------------------------
def Primitive2Conserved( W ):
    U = np.empty( 9 )

    U[0] = W[0]             # density
    U[1] = W[0]*W[1]        # momentum_x
    U[2] = W[0]*W[2]        # momentum_y
    U[3] = W[0]*W[3]        # momentum_z
    U[4] = ( W[4] ) / (gamma-1.0) + 0.5*W[0]*( W[1]**2.0 + W[2]**2.0 + W[3]**2.0 ) \
            + 0.5*(W[5]**2.0 + W[6]**2.0 + W[7]**2.0)
                            # energy
    U[5] = W[5]             # Bx
    U[6] = W[6]             # By
    U[7] = W[7]             # Bz
    U[8] = W[8]             # psi

    return U


# -------------------------------------------------------------------
# piecewise-linear data reconstruction
# -------------------------------------------------------------------
def DataReconstruction_PLM(U):
    W = np.empty((N, N, 9))
    L = np.empty((N, N, 9))
    R = np.empty((N, N, 9))

    # Conserved variables --> primitive variables
    for i in range(N):
        for j in range(N):
            W[i, j] = Conserved2Primitive(U[i, j])

    for i in range(1, N-1):
        for j in range(1, N-1):
            # Compute the left and right states of each cell
            slope_limited_x = ComputeLimitedSlope(W[i-1, j], W[i, j], W[i+1, j])
            slope_limited_y = ComputeLimitedSlope(W[i, j-1], W[i, j], W[i, j+1])

            # Get the face-centered variables
            L_x = W[i, j] - 0.5 * slope_limited_x
            R_x = W[i, j] + 0.5 * slope_limited_x
            L_y = W[i, j] - 0.5 * slope_limited_y
            R_y = W[i, j] + 0.5 * slope_limited_y

            # Ensure face-centered variables lie between nearby volume-averaged (~cell-centered) values
            L_x = np.maximum(L_x, np.minimum(W[i-1, j], W[i, j]))
            L_x = np.minimum(L_x, np.maximum(W[i-1, j], W[i, j]))
            R_x = 2.0 * W[i, j] - L_x

            R_x = np.maximum(R_x, np.minimum(W[i+1, j], W[i, j]))
            R_x = np.minimum(R_x, np.maximum(W[i+1, j], W[i, j]))
            L_x = 2.0 * W[i, j] - R_x

            L_y = np.maximum(L_y, np.minimum(W[i, j-1], W[i, j]))
            L_y = np.minimum(L_y, np.maximum(W[i, j-1], W[i, j]))
            R_y = 2.0 * W[i, j] - L_y

            R_y = np.maximum(R_y, np.minimum(W[i, j+1], W[i, j]))
            R_y = np.minimum(R_y, np.maximum(W[i, j+1], W[i, j]))
            L_y = 2.0 * W[i, j] - R_y

            # Combine x and y direction face-centered variables
            L[i, j] = 0.5 * (L_x + L_y)
            R[i, j] = 0.5 * (R_x + R_y)

            # Primitive variables --> conserved variables
            L[i, j] = Primitive2Conserved(L[i, j])
            R[i, j] = Primitive2Conserved(R[i, j])

    return L, R


# -------------------------------------------------------------------
# convert conserved variables to fluxes
# -------------------------------------------------------------------

# x-direction flux
def Conserved2Flux_x( U ):
    flux = np.empty( 9 )

    P = ComputeTotalPressure( U[0], U[1], U[2], U[3], U[4], U[5], U[6], U[7], U[8] )
    u = U[1] / U[0]
    v = U[2] / U[0]
    w = U[3] / U[0]

    flux[0] = U[1]
    flux[1] = U[0]*u*u + P - U[5]**2
    flux[2] = U[0]*u*v - U[5]*U[6] 
    flux[3] = U[0]*u*w - U[5]*U[7]
    flux[4] = (U[4] + P)*u - U[5]*np.dot([ U[5], U[6], U[7] ], [ u, v, w ] )
    flux[5] = 0
    flux[6] = u*U[6] - v*U[5]
    flux[7] = u*U[7] - w*U[5]
    flux[8] = 0.0
    
    return flux

# y-direction flux
def Conserved2Flux_y(U):
    flux = np.empty(9)
    P = ComputeTotalPressure(U[0], U[1], U[2], U[3], U[4], U[5], U[6], U[7], U[8])
    u = U[1] / U[0]
    v = U[2] / U[0]
    w = U[3] / U[0]

    flux[0] = U[2]
    flux[1] = U[0]*u*v - U[5]*U[6]
    flux[2] = U[0]*v*v + P - U[6]**2
    flux[3] = U[0]*v*w - U[6]*U[7]
    flux[4] = (U[4] + P)*v - U[6]*np.dot([U[5], U[6], U[7]], [u, v, w])
    flux[5] = v*U[5] - u*U[6]
    flux[6] = 0
    flux[7] = v*U[7] - w*U[6]
    flux[8] = 0.0

    return flux


# -------------------------------------------------------------------
# HLL Scheme
# -------------------------------------------------------------------
def HLL(L, R, direction='x'):
    rhoL_sqrt = L[0]**0.5
    rhoR_sqrt = R[0]**0.5

    # Compute the enthalpy of the left and right states: H = (E+P)/rho
    P_Lg = ComputePressure(L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7],L[8])
    P_Rg = ComputePressure(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8])
    P_L = ComputeTotalPressure(L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7],L[8])
    P_R = ComputeTotalPressure(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8])

    H_L = (L[4] + P_Lg) / L[0]
    H_R = (R[4] + P_Rg) / R[0]

    # Compute Roe-averaged values
    H = (rhoL_sqrt * H_L + rhoR_sqrt * H_R) / (rhoL_sqrt + rhoR_sqrt)
    u = (L[1] / rhoL_sqrt + R[1] / rhoR_sqrt) / (rhoL_sqrt + rhoR_sqrt)
    v = (L[2] / rhoL_sqrt + R[2] / rhoR_sqrt) / (rhoL_sqrt + rhoR_sqrt)
    w = (L[3] / rhoL_sqrt + R[3] / rhoR_sqrt) / (rhoL_sqrt + rhoR_sqrt)
    rho = rhoL_sqrt * rhoR_sqrt

    V2 = u * u + v * v + w * w
    p = rho * (H - 0.5 * V2) * ((gamma - 1.0) / gamma)
    B = [(rhoL_sqrt * L[5] + rhoR_sqrt * R[5]) / (rhoL_sqrt + rhoR_sqrt),
         (rhoL_sqrt * L[6] + rhoR_sqrt * R[6]) / (rhoL_sqrt + rhoR_sqrt),
         (rhoL_sqrt * L[7] + rhoR_sqrt * R[7]) / (rhoL_sqrt + rhoR_sqrt)]

    # Compute wave speeds
    a = (gamma * p / rho)**0.5
    b = ((B[0]**2.0 + B[1]**2.0 + B[2]**2.0) / rho)**0.5
    bx = (B[0]**2.0 / rho)**0.5

    ca = (B[0]**2.0 / rho)**0.5
    cf = ((a**2.0 + b**2.0 + ((a**2.0 + b**2.0)**2.0 - 4 * (a**2.0) * (bx**2.0))**0.5) / 2.0)**0.5
    cs = ((a**2.0 + b**2.0 - ((a**2.0 + b**2.0)**2.0 - 4 * (a**2.0) * (bx**2.0))**0.5) / 2.0)**0.5

    aL = (gamma * P_Lg / L[0])**0.5
    bL = ((L[5]**2.0 + L[6]**2.0 + L[7]**2.0) / L[0])**0.5
    bxL = (L[5]**2.0 / L[0])**0.5
    aR = (gamma * P_Rg / R[0])**0.5
    bR = ((R[5]**2.0 + R[6]**2.0 + R[7]**2.0) / R[0])**0.5
    bxR = (R[5]**2.0 / R[0])**0.5

    cL = ((aL**2 + bL**2 + ((aL**2 + bL**2)**2 - 4 * (aL**2) * (bxL**2))**0.5) / 2)**0.5
    cR = ((aR**2 + bR**2 + ((aR**2 + bR**2)**2 - 4 * (aR**2) * (bxR**2))**0.5) / 2)**0.5

    # Compute maximum information propagation speed
    sL = min(u, u + ca, u + cf, u + cs, u - ca, u - cf, u - cs, L[1] / (rhoL_sqrt)**2 - cL)
    sR = max(u, u + ca, u + cf, u + cs, u - ca, u - cf, u - cs, R[1] / (rhoR_sqrt)**2 + cR)

    SR = max(sR, 0)
    SL = min(sL, 0)

    if direction == 'x':
        flux_R = Conserved2Flux_x(R)
        flux_L = Conserved2Flux_x(L)
    else:
        flux_R = Conserved2Flux_y(R)
        flux_L = Conserved2Flux_y(L)

    flux_HLL = (SR * flux_L - SL * flux_R + SR * SL * (R - L)) / (SR - SL)
    if SL >= 0:
        flux = flux_L.copy()
    elif SL < 0 and SR > 0:
        flux = flux_HLL.copy()
    else:
        flux = flux_R.copy()

    return flux


# -------------------------------------------------------------------
# 定義 divergence !!
# -------------------------------------------------------------------
def ComputeDivergenceB(U, dx):
    bx = U[:, :, 5]
    by = U[:, :, 6]
    bz = U[:, :, 7]
    divB = np.zeros_like(bx)

    # 計算 bx 的散度
    divB[1:-1, 1:-1] += (bx[2:, 1:-1] - bx[1:-1, 1:-1]) / dx
    # 計算 by 的散度
    divB[1:-1, 1:-1] += (by[1:-1, 2:] - by[1:-1, 1:-1]) / dx
    # 計算 bz 的散度
    divB[1:-1, 1:-1] += (bz[1:-1, 1:-1] - bz[1:-1, :-2]) / dx

    return divB

# -------------------------------------------------------------------
# 定義 EGLM clean !!
# -------------------------------------------------------------------

# EGLM_hyperbolic
def hyperbolic_secondstep(flux, L, R):
    ch = 0.8       #目前只是亂設一個介於（0,1)的數字
    flux_hyp = flux.copy()
    b_xm = L[5] + 0.5*( R[5] - L[5]) - ( R[8] - L[8] ) / ( 2*ch )
    psi_m = L[8] + 0.5*( R[8] - L[8]) - 0.5*ch*( R[5] - L[5] ) 
    flux_hyp[8] = flux[8] + psi_m
    flux_hyp[5] = flux[5] + ch**2*b_xm

    return flux_hyp


# EGLM_mixed
def mixed_secondstep(flux, L, R):
    ch = 1.0
    flux_mixed = flux.copy()
    b_xm = L[5] + 0.5*( R[5] - L[5]) - ( R[8] - L[8] ) / ( 2*ch )
    psi_m = L[8] + 0.5*( R[8] - L[8]) - ch*( R[5] - L[5] ) / 2  
    flux_mixed[8] = flux[8] + ch**2*b_xm
    flux_mixed[5] = flux[5] + psi_m

    return flux_mixed



# -------------------------------------------------------------------
# 定義 update function !!
# -------------------------------------------------------------------

def update(frame):
    global t, U

    # For frame==0, just plot the initial condition
    if frame > 0:
        for step in range(nstep_per_image):
            # Set the boundary conditions
            BoundaryCondition(U)  # 這裡只傳遞一個參數 U

            # Estimate time-step from the CFL condition
            dt = 0.01
            #dt = ComputeTimestep(U, dx, dy, cfl, gamma, end_time, t)
            print("t = %13.7e --> %13.7e, dt = %13.7e" % (t, t+dt, dt))

            # Data reconstruction
            L, R = DataReconstruction_PLM(U)

            # Update the face-centered variables by 0.5*dt
            for i in range(1, N-1):
                for j in range(1, N-1):
                    flux_L_x = Conserved2Flux_x(L[i, j])
                    flux_R_x = Conserved2Flux_x(R[i, j])
                    flux_L_y = Conserved2Flux_y(L[i, j])
                    flux_R_y = Conserved2Flux_y(R[i, j])

                    dflux_x = 0.5 * dt / dx * (flux_R_x - flux_L_x)
                    dflux_y = 0.5 * dt / dy * (flux_R_y - flux_L_y)

                    L[i, j] -= dflux_x + dflux_y
                    R[i, j] -= dflux_x + dflux_y

            # Compute fluxes
            flux_x = np.zeros((N, N, 9))
            flux_y = np.zeros((N, N, 9))
            for i in range(nghost, N-nghost+1):
                for j in range(nghost, N-nghost+1):
                    flux_x[i, j] = HLL(R[i-1, j], L[i, j], direction='x')
                    flux_y[i, j] = HLL(R[i, j-1], L[i, j], direction='y')

            #EGLM_hyperboliic
#                    flux_x[i, j] = hyperbolic_secondstep(flux_x[i, j], R[i-1, j], L[i, j])
#                    flux_y[i, j] = hyperbolic_secondstep(flux_x[i, j], R[i, j-1], L[i, j])

            # Update the conserved variables
            for i in range(nghost, N-nghost):
                for j in range(nghost, N-nghost):
                    U[i, j, :] -= dt / dx * (flux_x[i+1, j, :] - flux_x[i, j, :])
                    U[i, j, :] -= dt / dy * (flux_y[i, j+1, :] - flux_y[i, j, :])

            # Update time
            t = t + dt

            # need to update the psi for EGLM_mixed  (hyperbolic dont need)
#            U[:,8] = c_d*U[:,8]


#           check div B
#           Compute and store divergence of B
            divB = ComputeDivergenceB(U, dx)
            divB_values.append(np.sum(divB))
            print("t = ", t, ", mean |divB| = ", np.sum(divB))
            if ( t >= end_time ):
                anim.event_source.stop()
                break


    # Plot
    d = U[nghost:N - nghost, nghost:N - nghost, 0]
    cax.set_array(d)
    ax.set_title('t = %6.3f' % (t))

    return cax, text


#--------------------------------------------------------------------
# parameters
#--------------------------------------------------------------------
# constants
L        = 1.0       # 1-D computational domain size
N_In     = 100       # number of computing cells
cfl      = 0.475       # Courant factor
nghost   = 2         # number of ghost zones
gamma    = 5.0/3.0   # ratio of specific heats
end_time = 10    # simulation time
c_h = cfl
c_d = 0.5 #0~1
divB_values = []


# derived constants
N  = N_In + 2*nghost    # total number of cells including ghost zones
dx = L/N_In             # spatial resolution
dy = L/N_In             # spatial resolution


# plotting parameters
plot_width      = 10.0  # width of the plotting domain
nstep_per_image = 10     # plotting frequency


#--------------------------------------------------------------------
# main
#--------------------------------------------------------------------
# set initial condition
t = 0.0
U = np.zeros((N, N, 9))  # Conserved variables
x = np.linspace(-nghost*dx, L + nghost*dx, N)
y = np.linspace(-nghost*dy, L + nghost*dy, N)
#x = np.linspace(0, 1, N)
#y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)
for i in range(N):
    for j in range(N):
        U[i, j] = InitialCondition(X[i, j], Y[i, j])

BoundaryCondition(U)

# create figure
fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
cax = ax.imshow(U[nghost:N - nghost, nghost:N - nghost, 0], origin='lower', extent=[0, L, 0, L])
fig.colorbar(cax)
text = ax.text(0.6, 0.9, '', fontsize=12, color='black', ha='center', va='center', transform=ax.transAxes)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.tick_params(top=True, right=True, labeltop=True, labelright=True)



# -------------------------------------------------------------------
# initialize animation
# -------------------------------------------------------------------

def init():
    cax.set_array(np.zeros((N - 2 * nghost, N - 2 * nghost)))
    text.set_text('')
    return cax, text

# create movie
nframe = 99999999  # arbitrarily large
anim = animation.FuncAnimation(fig, func=update, init_func=init, frames=nframe, interval=200, repeat=False)
plt.show()


# plot divergence
divB_values = np.array(divB_values)
# print("time_values:", time_values)
# print("divB_values:", divB_values)
plt.figure(figsize=(10, 6))
plt.plot( divB_values, marker='o', linestyle='-', color='b', label='divergence of B')
plt.xlabel('Time t')
plt.ylabel('Divergence of B')
plt.title('Divergence of B vs Time')
plt.grid(True)
plt.legend()
plt.yscale('linear')  # 明確設定 y 軸為線性尺度
plt.xscale('linear')  # 明確設定 y 軸為線性尺度
plt.tight_layout()
plt.show()