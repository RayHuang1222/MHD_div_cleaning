#--------------------------------------------------------------------
# Test Brio and Wu problem with the MUSCL-Hancock scheme
#--------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank() # my rank
nrank = comm.Get_size() # total number of processes
assert nrank==2, "Only works for 2 processes for now"

#--------------------------------------------------------------------
# parameters
#--------------------------------------------------------------------
# constants
L        = 1.0       # 1-D computational domain size
N_In     = 200       # number of computing cells
n_In     = int(N_In/nrank)# number of computing cells per process
cfl      = 0.475       # Courant factor
nghost   = 2         # number of ghost zones
gamma    = 5.0/3.0   # ratio of specific heats
end_time = 0.01      # simulation time
c_h = cfl
c_d = 0.5 #0~1
time_values = []
divB_values = []


# derived constants
N  = N_In + 2*nghost    # total number of cells including ghost zones
n  = n_In + 2*nghost    # total number of cells including ghost zones per process
dx = L/N_In             # spatial resolution

# mpi
des = (rank+1)%2 #target rank
src = (rank+1)%2 #source rank

# plotting parameters
nstep_per_image = 1     # plotting frequency

# -------------------------------------------------------------------
# define initial condition
# -------------------------------------------------------------------
def InitialCondition( x ):
#  Brio and Wu Shock Tube
    if ( x < 0.5*L ):
        d = 1.0  # density
        u = 0.0  # velocity x
        v = 0.0
        w = 0.0
        bx = 1.0
        by = 1.0
        bz = 0.0
        P = 1.0  # gas pressure
        E = ( P ) /(gamma-1.0) + 0.5*d*(u**2.0+v**2+w**2) + 0.5*(bx**2+by**2+bz**2)    # total energy density
    else:
        d = 0.125  # density
        u = 0.0  # velocity x
        v = 0.0
        w = 0.0
        bx = 1.0
        by = -1.0
        bz = 0.0
        P = 0.1  # gas pressure
        E = ( P ) /(gamma-1.0) + 0.5*d*(u**2.0+v**2+w**2) + 0.5*(bx**2+by**2+bz**2)     # total energy density
        
#  conserved variables [0/1/2/3/4] <--> [density/momentum x/momentum y/momentum z/energy/B-field x/B-field y/B-field z/psi]
    return np.array( [d, d*u, d*v, d*w, E, bx, by, bz, 0.0] )

# -------------------------------------------------------------------
# define boundary condition by setting ghost zones
# -------------------------------------------------------------------
def BoundaryCondition( U, bd_r, bd_l ):
#  outflow
   if rank == 0:
      U[0:nghost]   = U[nghost]
      U[n-nghost:n] = bd_r
   else:
      U[0:nghost]   = bd_l
      U[n-nghost:n] = U[n-nghost-1]

#    U[0:nghost]   = U[nghost]
#    U[N-nghost:N] = U[N-nghost-1]

#  Dirichlet for Brio and Wu Shock Tube
    #U[0:nghost]   = [1.0  , 0.0, 0.0, 0.0, 2.28125, 0.75, 1.0, 0.0, 0.0]
    #U[N-nghost:N] = [0.125, 0.0, 0.0, 0.0, 0.93125, 0.75, -1.0, 0.0, 0.0]


# -------------------------------------------------------------------
# compute pressure
# -------------------------------------------------------------------
def ComputePressure( d, px, py, pz, e, bx, by, bz, psi ):
    B = (bx**2.0 + by**2.0 + bz**2.0)**0.5
    P = (gamma-1.0)*( e - 0.5*(px**2.0 + py**2.0 + pz**2.0)/d - 0.5*B**2.0 ) 
    assert np.all( P > 0 ), "negative pressure !!"
    return P

def ComputeTotalPressure( d, px, py, pz, e, bx, by, bz, psi ):
    B = (bx**2.0 + by**2.0 + bz**2.0)**0.5
    P = (gamma-1.0)*( e - 0.5*(px**2.0 + py**2.0 + pz**2.0)/d - 0.5*B**2.0 ) + 0.5*B**2.0
    assert np.all( P > 0 ), "negative pressure !!"
    return P

# -------------------------------------------------------------------
# compute time-step by the CFL condition
# -------------------------------------------------------------------
def ComputeTimestep( U ):
    P = ComputePressure( U[:,0], U[:,1], U[:,2], U[:,3], U[:,4], U[:,5], U[:,6], U[:,7], U[:,8] )
    u = np.abs( U[:,1]/U[:,0] )
    v = np.abs( U[:,2]/U[:,0] )
    w = np.abs( U[:,3]/U[:,0] )

    a = ( gamma*P/U[:,0])**0.5
    b = ( ( U[:,5]**2.0+U[:,6]**2.0+U[:,5]**7.0 ) / U[:,0] )**0.5
    bx = ( U[:,5]**2.0 / U[:,0] )**0.5
    

    cf = (( a**2.0+b**2.0 + ( ( a**2.0+b**2.0 )**2.0 - 4*(a**2.0)*(bx**2.0) )**0.5 ) / 2.0 )**0.5
    
#  maximum information speed in 3D
    max_info_speed = np.amax( u + cf )
    dt_cfl         = cfl*dx/max_info_speed
    tsend = np.array([dt_cfl])
    trecv = np.zeros(1)
    if rank == 0:
        comm.Send(tsend, des, tag = 1000)
        comm.Recv(trecv, src, tag = 1001)
    else:
        comm.Recv(trecv, src, tag = 1000)
        comm.Send(tsend, des, tag = 1001)
    dt_cfl2 = trecv[0]
    dt_end = end_time-t


    return  min( dt_cfl, dt_cfl2, dt_end )


# -------------------------------------------------------------------
# compute limited slope
# -------------------------------------------------------------------
def ComputeLimitedSlope( L, C, R ):
#  compute the left and right slopes
    slope_L = C - L
    slope_R = R - C

#  apply the van-Leer limiter
    slope_LR      = slope_L*slope_R
    slope_limited = np.where( slope_LR>0.0, 2.0*slope_LR/(slope_L+slope_R), 0.0 )

#  apply the MinMod limiter
#    slope_limited = np.zeros( len(slope_L) )
#    for i in range(0, len(slope_limited)):
#        if slope_L[i]*slope_R[i] >= 0:
#            slope_limited[i] = np.sign(slope_L[i])*min(np.abs(slope_L[i]), np.abs(slope_R[i]))
#        else:
#            slope_limited[i] = 0.0
            
    return slope_limited

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
def DataReconstruction_PLM( U ):

#  allocate memory
    W = np.empty( (n,9) )
    L = np.empty( (n,9) )
    R = np.empty( (n,9) )

#  conserved variables --> primitive variables
    for j in range( n ):
        W[j] = Conserved2Primitive( U[j] )

    for j in range( 1, n-1 ):
#     compute the left and right states of each cell
        slope_limited = ComputeLimitedSlope( W[j-1], W[j], W[j+1] )

#     get the face-centered variables
        L[j] = W[j] - 0.5*slope_limited
        R[j] = W[j] + 0.5*slope_limited

#     ensure face-centered variables lie between nearby volume-averaged (~cell-centered) values
        L[j] = np.maximum( L[j], np.minimum( W[j-1], W[j] ) )
        L[j] = np.minimum( L[j], np.maximum( W[j-1], W[j] ) )
        R[j] = 2.0*W[j] - L[j]

        R[j] = np.maximum( R[j], np.minimum( W[j+1], W[j] ) )
        R[j] = np.minimum( R[j], np.maximum( W[j+1], W[j] ) )
        L[j] = 2.0*W[j] - R[j]

#     primitive variables --> conserved variables
        L[j] = Primitive2Conserved( L[j] )
        R[j] = Primitive2Conserved( R[j] )

    return L, R

# -------------------------------------------------------------------
# convert conserved variables to fluxes
# -------------------------------------------------------------------
def Conserved2Flux( U ):
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
    flux[5] = u*U[5] - u*U[5]
    flux[6] = u*U[6] - v*U[5]
    flux[7] = u*U[7] - w*U[5]
    flux[8] = 0.0
    
    return flux
# -------------------------------------------------------------------
# HLL Scheme
# -------------------------------------------------------------------
def HLL( L, R ):

    rhoL_sqrt = L[0]**0.5
    rhoR_sqrt = R[0]**0.5

#  compute the enthalpy of the left and right states: H = (E+P)/rho
    P_Lg = ComputePressure( L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7], L[8] )
    P_Rg = ComputePressure( R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8] )
    P_L  = ComputeTotalPressure( L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7], L[8] )
    P_R  = ComputeTotalPressure( R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8] )

    H_L = ( L[4] + P_Lg )/L[0]
    H_R = ( R[4] + P_Rg )/R[0]

#  compute Roe-averaged values
    H  = ( rhoL_sqrt*H_L  + rhoR_sqrt*H_R  ) / ( rhoL_sqrt + rhoR_sqrt )
    u  = ( L[1]/rhoL_sqrt + R[1]/rhoR_sqrt ) / ( rhoL_sqrt + rhoR_sqrt )
    v  = ( L[2]/rhoL_sqrt + R[2]/rhoR_sqrt ) / ( rhoL_sqrt + rhoR_sqrt )
    w  = ( L[3]/rhoL_sqrt + R[3]/rhoR_sqrt ) / ( rhoL_sqrt + rhoR_sqrt )
    rho = rhoL_sqrt*rhoR_sqrt

    V2 = u*u + v*v + w*w
    p = rho*(H - 0.5*V2)*( ( gamma-1.0 ) / (gamma) )
    B = [( rhoL_sqrt*L[5] + rhoR_sqrt*R[5]  ) / ( rhoL_sqrt + rhoR_sqrt ),  \
         ( rhoL_sqrt*L[6] + rhoR_sqrt*R[6]  ) / ( rhoL_sqrt + rhoR_sqrt ),  \
         ( rhoL_sqrt*L[7] + rhoR_sqrt*R[7]  ) / ( rhoL_sqrt + rhoR_sqrt )]

#  compute wave speeds
    a = ( gamma*p/rho )**0.5
    b = ( ( B[0]**2.0+B[1]**2.0+B[2]**2.0 ) / rho )**0.5
    bx = ( B[0]**2.0 / rho )**0.5

    ca = ( B[0]**2.0 / rho)**0.5
    cf = (( a**2.0+b**2.0 + ( ( a**2.0+b**2.0 )**2.0 - 4*(a**2.0)*(bx**2.0) )**0.5 ) / 2.0 )**0.5
    cs = (( a**2.0+b**2.0 - ( ( a**2.0+b**2.0 )**2.0 - 4*(a**2.0)*(bx**2.0) )**0.5 ) / 2.0 )**0.5

    aL = ( gamma*P_Lg/L[0] )**0.5
    bL = ( ( L[5]**2.0+L[6]**2.0+L[7]**2.0 ) / L[0] )**0.5
    bxL = ( L[5]**2.0 / L[0] )**0.5
    aR = ( gamma*P_Rg/R[0] )**0.5
    bR = ( ( R[5]**2.0+R[6]**2.0+R[7]**2.0 ) / R[0] )**0.5
    bxR = ( R[5]**2.0 / R[0] )**0.5

    cL = (( aL**2+bL**2 + ( ( aL**2+bL**2 )**2 - 4*(aL**2)*(bxL**2) )**0.5 ) / 2 )**0.5
    cR = (( aR**2+bR**2 + ( ( aR**2+bR**2 )**2 - 4*(aR**2)*(bxR**2) )**0.5 ) / 2 )**0.5

#  compute maximum information propagation speed
    sL = min( u, u+ca, u+cf, u+cs, u-ca, u-cf, u-cs, L[1]/(rhoL_sqrt)**2 - cL )
    sR = max( u, u+ca, u+cf, u+cs, u-ca, u-cf, u-cs, R[1]/(rhoR_sqrt)**2 + cR )

    SR = max( sR, 0 )
    SL = min( sL, 0 )

    flux_R = Conserved2Flux( R )
    flux_L = Conserved2Flux( L )

    flux_HLL = ( SR*flux_L - SL*flux_R + SR*SL*( R-L ) ) / ( SR-SL )
    if SL >= 0:
        flux = flux_L.copy()
    elif SL < 0 and SR > 0 :
        flux = flux_HLL.copy()
    else:
        flux = flux_R.copy()

    return flux

# mixed GLM_method
def mixed_secondstep(flux, L, R):
    global rank
    ch = 1.0
    flux_mixed = flux.copy()
    b_xm = L[5] + 0.5*( R[5] - L[5]) - ( R[8] - L[8] ) / ( 2*ch )
    psi_m = L[8] + 0.5*( R[8] - L[8]) - ch*( R[5] - L[5] ) / 2  
    flux_mixed[8] = flux[8] + ch**2*b_xm
    flux_mixed[5] = flux[5] + psi_m
#    print(rank,flux[5],psi_m)

    return flux_mixed

def hyperbolic_secondstep(flux, L, R):
    ch = 0.8       #目前只是亂設一個（0,1)的數字
    flux_hyp = flux.copy()
    b_xm = L[5] + 0.5*( R[5] - L[5]) - ( R[8] - L[8] ) / ( 2*ch )
    psi_m = L[8] + 0.5*( R[8] - L[8]) - 0.5*ch*( R[5] - L[5] ) 
    flux_hyp[8] = flux[8] + psi_m
    flux_hyp[5] = flux[5] + ch**2*b_xm

    return flux_hyp


# compute divergence
def ComputeDivergenceB(U, dx):
    bx = U[:, 5]
    by = U[:, 6]
    bz = U[:, 7]
    divB = np.zeros_like(bx)
    divB[:-1] = (bx[1:] - bx[:-1]) / dx + (by[1:] - by[:-1]) / dx + (bz[1:] - bz[:-1]) / dx

    return divB

# -------------------------------------------------------------------
# initialize animation
# -------------------------------------------------------------------
def init():
    line_d.set_xdata( x )
    line_u.set_xdata( x )
    line_v.set_xdata( x )
    line_bx.set_xdata( x )
    # line_by.set_xdata( x )
    # line_bz.set_xdata( x )
    line_p.set_xdata( x )
    return line_d, line_u, line_v, line_bx, line_p


def update( frame ):
    global t, U

    # split the cells into two halfs
    u = np.empty((n,9))
    if rank == 0:
        u = U[:n]
        u[0:nghost] = U[nghost]
    else:
        u = U[-n:]
        u[n-nghost:n] = U[n-nghost-1]
    bd_r = u[n-nghost:n]
    bd_l = u[0:nghost]
    it = 0

#  for frame==0, just plot the initial condition
    if frame > 0:
        for step in range( nstep_per_image ):
            it += 1

#        set the boundary conditions
            BoundaryCondition( u, bd_r, bd_l )

#        estimate time-step from the CFL condition
            dt = ComputeTimestep( u )
            # c_p = np.sqrt(-dt*c_h**2/np.ln(c_d))
            print( "t = %13.7e --> %13.7e, dt = %13.7e" % (t,t+dt,dt) )

#        data reconstruction
            L, R = DataReconstruction_PLM( u )

#        update the face-centered variables by 0.5*dt
            for j in range( 1, n-1 ):
                flux_L = Conserved2Flux( L[j] )
                flux_R = Conserved2Flux( R[j] )
                dflux  = 0.5*dt/dx*( flux_R - flux_L )
                L[j]  -= dflux
                R[j]  -= dflux

#        compute fluxes
            flux = np.empty( (n,9) )
            for j in range( nghost, n-nghost+1 ):
#           R[j-1] is the LEFT state at the j+1/2 inteface
#           first use HLL solver 
                flux[j] = HLL(R[j-1], L[j])

#           1.mixed correction
                flux[j] = mixed_secondstep(flux[j], R[j-1] ,L[j])
                
#           2.hyperbolic correction
#                flux[j] = hyperbolic_secondstep(flux[j], R[j-1] ,L[j])
                
#           update the volume-averaged input variables by dt
            u[nghost:n-nghost] -= dt/dx*( flux[nghost+1:n-nghost+1] - flux[nghost:n-nghost] )
            
#           update the psi for 1.mixed  (hyperbolic dont need)
            u[8] = c_d*u[8]

#           send boundary data            
            recv = np.empty((nghost,9))
            if rank == 0:
                send = u[n-nghost-nghost:n-nghost]
                comm.Send(send, des, tag = it)
                comm.Recv(recv, src, tag = it+1)
                bd_r = recv
            else:
                send = u[nghost:nghost+nghost]
                comm.Recv(recv, src, tag = it)
                comm.Send(send, des, tag = it+1)
                bd_l = recv

#           update time
            t = t + dt
#           Compute and store divergence of B
            divB = ComputeDivergenceB(u, dx)
#            print("Sum divergence of B at",t,"is",np.sum((divB)))
            if ( t >= end_time ):
                anim.event_source.stop()
                break

        # Compute divergence of B
        divB = ComputeDivergenceB(u, dx)
        time_values.append(t)
        divB_values.append(np.abs(sum(divB)))
#        print("Sum divergence of B at t = %f is %f" % (t, np.sum(divB)))

# combine data
    Recv = np.empty((n_In,9))
    if rank == 0:
        Send = u[nghost:n-nghost]
        comm.Send(Send, des, tag = 0)
        comm.Recv(Recv, src, tag = 1)
        U[nghost:int(N/2)] = u[nghost:n-nghost]
        U[int(N/2):N-nghost] = Recv
    else:
        Send = u[nghost:n-nghost]
        comm.Recv(Recv, src, tag = 0)
        comm.Send(Send, des, tag = 1)
        U[nghost:int(N/2)] = Recv
        U[int(N/2):N-nghost] = u[nghost:n-nghost]
            
            
    #  plot
    d  = U[nghost:N-nghost,0]
    u  = U[nghost:N-nghost,1] / U[nghost:N-nghost,0]
    v  = U[nghost:N-nghost,2] / U[nghost:N-nghost,0]
    bx = U[nghost:N-nghost,5]
    by = U[nghost:N-nghost,6]
    bz = U[nghost:N-nghost,7]
    P  = ComputePressure( U[nghost:N-nghost,0], U[nghost:N-nghost,1], U[nghost:N-nghost,2], U[nghost:N-nghost,3],
                          U[nghost:N-nghost,4], U[nghost:N-nghost,5], U[nghost:N-nghost,6], U[nghost:N-nghost,7],
                          U[nghost:N-nghost,8])
    line_d.set_ydata( d )
    line_u.set_ydata( u )
    line_v.set_ydata( v )
    line_bx.set_ydata( bx )
    # line_by.set_ydata( by )
    # line_bz.set_ydata( bz )
    line_p.set_ydata( P )
#  ax[0].legend( loc='upper right', fontsize=12 )
#  ax[1].legend( loc='upper right', fontsize=12 )
#  ax[2].legend( loc='upper right', fontsize=12 )
    ax[0].set_title( 't = %6.3f' % (t) )
    return line_d, line_u, line_v, line_bx, line_by, line_bz, line_p
#    return line_d, line_u, line_v, line_bx, line_p


## Load analytical solution (strong_shock, depend on your path)
data_strong_shock = np.loadtxt("BrioWu.txt") 
x_analytical = data_strong_shock[:,0]
d_analytical = data_strong_shock[:,1]   
u_analytical = data_strong_shock[:,2]   
v_analytical = data_strong_shock[:,3]   
bx_analytical = data_strong_shock[:,5]  
by_analytical = data_strong_shock[:,6]  
bz_analytical = data_strong_shock[:,7]  
P_analytical = data_strong_shock[:,8]  


#--------------------------------------------------------------------
# main
#--------------------------------------------------------------------
# set initial condition
t = 0.0
x = np.empty( N_In )
U = np.empty( (N,9 ) )
for j in range( N_In ):
    x[j] = (j+0.5)*dx    # cell-centered coordinates
    U[j+nghost] = InitialCondition( x[j] )
    
#BoundaryCondition( U )

# create figure
fig, ax = plt.subplots( 7, 1, sharex=True, sharey=False, figsize=(8,12) )
fig.subplots_adjust( hspace=0.15, wspace=0.0 )
#fig.set_size_inches( 3.3, 12.8 )
#line_d_ref,  = ax[0].plot( r_ref, Rho_ref, ls='--', markeredgecolor='r', markersize=0.5 )
#line_u_ref,  = ax[1].plot( r_ref, Vx_ref, ls='--', markeredgecolor='r', markersize=0.5 )
#line_v_ref,  = ax[2].plot( r_ref, Vy_ref, ls='--', markeredgecolor='r', markersize=0.5 )
#line_by_ref, = ax[3].plot( r_ref, By_ref, ls='--', markeredgecolor='r', markersize=0.5)
#line_p_ref,  = ax[4].plot( r_ref, Pres_ref, ls='--', markeredgecolor='r', markersize=0.5 )
line_d,  = ax[0].plot( [], [], 'r-o', markeredgecolor='k', markersize=3 )
line_u,  = ax[1].plot( [], [], 'b-o', markeredgecolor='k', markersize=3 )
line_v,  = ax[2].plot( [], [], 'g-o', markeredgecolor='k', markersize=3 )
line_bx, = ax[3].plot( [], [], 'g-o', markeredgecolor='k', markersize=3 )
line_by, = ax[4].plot( [], [], 'g-o', markeredgecolor='k', markersize=3 )
line_bz, = ax[5].plot( [], [], 'g-o', markeredgecolor='k', markersize=3 )
line_p,  = ax[6].plot( [], [], 'k-o', markeredgecolor='k', markersize=3 )
ax[2].set_xlabel( 'x' )
ax[0].set_ylabel( 'Density' )
ax[1].set_ylabel( r'$v_x$' )
ax[2].set_ylabel( r'$v_y$' )
ax[3].set_ylabel( r'$B_x$' )
ax[4].set_ylabel( r'$B_y$' )
ax[5].set_ylabel( r'$B_z$' )
ax[6].set_ylabel( 'Pressure' )
ax[0].set_xlim( 0.30, L-0.25 )
ax[0].set_ylim( +0.0, 1.2 )
ax[1].set_ylim( -1.0, 1.0 )
ax[2].set_ylim( -2.0, 2.0 )
ax[3].set_ylim( -1.5, 1.5 )
ax[4].set_ylim( -1.5, 1.5 )
ax[5].set_ylim( -1.5, 1.5 )
ax[6].set_ylim( 0.0, 2.5)

# Plot analytical solution
ax[0].plot(x_analytical, d_analytical, 'k-', lw=1.5, label='Analytical')
ax[1].plot(x_analytical, u_analytical, 'k-', lw=1.5, label='Analytical')    
ax[2].plot(x_analytical, v_analytical, 'k-', lw=1.5, label='Analytical')
ax[3].plot(x_analytical, bx_analytical, 'k-', lw=1.5, label='Analytical')
ax[4].plot(x_analytical, by_analytical, 'k-', lw=1.5, label='Analytical')
ax[5].plot(x_analytical, bz_analytical, 'k-', lw=1.5, label='Analytical')
ax[6].plot(x_analytical, P_analytical, 'k-', lw=1.5, label='Analytical')


# create movie
nframe = 99999999 # arbitrarily large
anim   = animation.FuncAnimation( fig, func=update, init_func=init,
                                  frames=nframe, interval=200, repeat=False )
plt.show()



# To see whether the divergence B have decreased
time_values = np.array(time_values)
divB_values = np.array(divB_values)
# print("time_values:", time_values)
# print("divB_values:", divB_values)
plt.figure(figsize=(10, 6))
plt.plot(time_values, divB_values, marker='o', linestyle='-', color='b', label='divergence of B')
plt.xlabel('Time t')
plt.ylabel('Divergence of B')
plt.title('Divergence of B vs Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
