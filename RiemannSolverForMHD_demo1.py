import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        bx = 0.0
        by = 1.0
        bz = 0.0
        P = 1.0  # gas pressure
        E = ( P ) /(gamma-1.0) + 0.5*d*(u**2.0+v**2+w**2) + 0.5*(bx**2+by**2+bz**2)    # energy density
    else:
        d = 0.125  # density
        u = 0.0  # velocity x
        v = 0.0
        w = 0.0
        bx = 0.0
        by = -1.0
        bz = 0.0
        P = 0.1  # gas pressure
        E = ( P ) /(gamma-1.0) + 0.5*d*(u**2.0+v**2+w**2) + 0.5*(bx**2+by**2+bz**2)     # energy density
        
#  conserved variables [0/1/2/3/4] <--> [density/momentum x/momentum y/momentum z/energy/B-field x/B-field y/B-field z]
    return np.array( [d, d*u, d*v, d*w, E, bx, by, bz] )

# -------------------------------------------------------------------
# define boundary condition by setting ghost zones
# -------------------------------------------------------------------
def BoundaryCondition( U ):
#  outflow
    U[0:nghost]   = U[nghost]
    U[N-nghost:N] = U[N-nghost-1]

#  Dirichlet
    #U[0:nghost]   = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0]
    #U[N-nghost:N] =  [1.0, 0.0, 0.0, 0.0, 0.6875, 0.0, -1.0, 0.0]

# -------------------------------------------------------------------
# compute pressure
# -------------------------------------------------------------------
def ComputePressure( d, px, py, pz, e, bx, by, bz ):
    B = (bx**2+by**2+bz**2)**0.5
    P = (gamma-1.0)*( e - 0.5*(px**2.0 + py**2.0 + pz**2.0)/d - 0.5*B**2 ) 
    assert np.all( P > 0 ), "negative pressure !!"
    return P

def ComputeTotalPressure( d, px, py, pz, e, bx, by, bz ):
    B = (bx**2+by**2+bz**2)**0.5
    P = (gamma-1.0)*( e - 0.5*(px**2.0 + py**2.0 + pz**2.0)/d - 0.5*B**2 ) + 0.5*B**2
    assert np.all( P > 0 ), "negative pressure !!"
    return P
# -------------------------------------------------------------------
# compute time-step by the CFL condition
# -------------------------------------------------------------------
def ComputeTimestep( U ):
    P = ComputePressure( U[:,0], U[:,1], U[:,2], U[:,3], U[:,4], U[:,5], U[:,6], U[:,7] )
    a = ( gamma*P/U[:,0] )**0.5
    u = np.abs( U[:,1]/U[:,0] )
    v = np.abs( U[:,2]/U[:,0] )
    w = np.abs( U[:,3]/U[:,0] )

#  maximum information speed in 3D
    max_info_speed = np.amax( u + a )
    dt_cfl         = cfl*dx/max_info_speed
    dt_end         = end_time - t

    return min( dt_cfl, dt_end )


# -------------------------------------------------------------------
# compute limited slope
# -------------------------------------------------------------------
def ComputeLimitedSlope( L, C, R ):
#  compute the left and right slopes
    slope_L = C - L
    slope_R = R - C

#  apply the van-Leer limiter
    #slope_LR      = slope_L*slope_R
    #slope_limited = np.where( slope_LR>0.0, 2.0*slope_LR/(slope_L+slope_R), 0.0 )

#  apply the MinMod limiter
    slope_limited = np.zeros( len(slope_L) )
    for i in range(0, len(slope_limited)):
        if slope_L[i]*slope_R[i] >= 0:
            slope_limited[i] = np.sign(slope_L[i])*min(np.abs(slope_L[i]), np.abs(slope_R[i]))
        else:
            slope_limited[i] = 0.0
            
    return slope_limited

# -------------------------------------------------------------------
# convert conserved variables to primitive variables
# -------------------------------------------------------------------
def Conserved2Primitive( U ):
    W = np.empty( 8 )

    W[0] = U[0]
    W[1] = U[1]/U[0]
    W[2] = U[2]/U[0]
    W[3] = U[3]/U[0]
    W[4] = ComputePressure( U[0], U[1], U[2], U[3], U[4], U[5], U[6], U[7] )
    W[5] = U[5]
    W[6] = U[6]
    W[7] = U[7]
    
    return W

# -------------------------------------------------------------------
# convert primitive variables to conserved variables
# -------------------------------------------------------------------
def Primitive2Conserved( W ):
    U = np.empty( 8 )

    U[0] = W[0]
    U[1] = W[0]*W[1]
    U[2] = W[0]*W[2]
    U[3] = W[0]*W[3]
    U[4] = ( W[4] ) / (gamma-1.0) + 0.5*W[0]*( W[1]**2.0 + W[2]**2.0 + W[3]**2.0 ) \
            + 0.5*(W[5]**2+W[6]**2+W[7]**2)
    U[5] = W[5]
    U[6] = W[6]
    U[7] = W[7]

    return U

# -------------------------------------------------------------------
# piecewise-linear data reconstruction
# -------------------------------------------------------------------
def DataReconstruction_PLM( U ):

#  allocate memory
    W = np.empty( (N,8) )
    L = np.empty( (N,8) )
    R = np.empty( (N,8) )

#  conserved variables --> primitive variables
    for j in range( N ):
        W[j] = Conserved2Primitive( U[j] )

    for j in range( 1, N-1 ):
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
    flux = np.empty( 8 )

    P = ComputeTotalPressure( U[0], U[1], U[2], U[3], U[4], U[5] , U[6] , U[7]  )
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
    
    return flux

# -------------------------------------------------------------------
# HLLC Scheme
# -------------------------------------------------------------------
def HLLC( L, R ):

    rhoL_sqrt = L[0]**0.5
    rhoR_sqrt = R[0]**0.5

#  compute the enthalpy of the left and right states: H = (E+P)/rho
    P_Lg = ComputePressure( L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7] )
    P_Rg = ComputePressure( R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7] )
    P_L = ComputeTotalPressure( L[0], L[1], L[2], L[3], L[4], L[5], L[6], L[7] )
    P_R = ComputeTotalPressure( R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7] )

    H_L = ( L[4] + P_Lg )/L[0]
    H_R = ( R[4] + P_Rg )/R[0]

#  compute sound speed
    H  = ( rhoL_sqrt*H_L  + rhoR_sqrt*H_R  ) / ( rhoL_sqrt + rhoR_sqrt )
    u  = ( L[1]/rhoL_sqrt + R[1]/rhoR_sqrt ) / ( rhoL_sqrt + rhoR_sqrt )
    v  = ( L[2]/rhoL_sqrt + R[2]/rhoR_sqrt ) / ( rhoL_sqrt + rhoR_sqrt )
    w  = ( L[3]/rhoL_sqrt + R[3]/rhoR_sqrt ) / ( rhoL_sqrt + rhoR_sqrt )
    V2 = u*u + v*v + w*w
    a  = ( (gamma-1.0)*(H - 0.5*V2) )**0.5

    cR = ( gamma*P_L/L[0] )**0.5
    cL = ( gamma*P_R/R[0] )**0.5

#  compute maximum information propagation speed
    bL = min( u-a, L[1]/(rhoL_sqrt)**2 - cL )
    bR = max( u+a, R[1]/(rhoR_sqrt)**2 + cR )

    SR = max( bR, 0 )
    SL = min( bL, 0 )

#  compute the interface speed
    uR = R[1]/(rhoR_sqrt)**2
    uL = L[1]/(rhoL_sqrt)**2
    vR = R[2]/(rhoR_sqrt)**2
    vL = L[2]/(rhoL_sqrt)**2
    wR = R[3]/(rhoR_sqrt)**2
    wL = L[3]/(rhoL_sqrt)**2


#  compute the interface magnetic field
    flux_R = Conserved2Flux( R )
    flux_L = Conserved2Flux( L )
    U_HLL = ( SR*R - SL*L - ( flux_R - flux_L ) ) / ( SR - SL )

    SM = ( P_R - P_L + L[0]*uL*( SL - uL ) - R[0]*uR*( SR - uR ) - R[5]**2 + L[5]**2 ) / ( L[0]*( SL - uL ) - R[0]*( SR - uR ) )
    P_LM = L[0]*( SL - uL )*( SM - uL ) + P_L - L[5]**2 + L[5]**2
    P_RM = R[0]*( SR - uR )*( SM - uR ) + P_R - R[5]**2 + R[5]**2

    if SL >= 0:
        U_HLL = L.copy()
    elif SL < 0 and SR > 0 :
        U_HLL = U_HLL
    else:
        U_HLL = R.copy()

    B_xM = U_HLL[5] #Bx = const in 1D
    B_yM = U_HLL[6]
    B_zM = U_HLL[7]
    B_L = [L[5], L[6], L[7]]
    B_R = [R[5], R[6], R[7]]
    B_LM = [B_xM, B_yM, B_zM]
    B_RM = [B_xM, B_yM, B_zM]

#  compute U star
    U_LM = np.empty( 8 )
    U_RM = np.empty( 8 )

    U_RM[0] = R[0]*( SR-uR ) / ( SR - SM )
    U_RM[1] = U_RM[0]*SM
    U_RM[2] = R[0]*vR*( SR - uR ) / ( SR - SM ) - ( B_xM*B_yM - R[5]*R[6] ) / ( SR - SM )
    U_RM[3] = R[0]*wR*( SR - uR ) / ( SR - SM ) - ( B_xM*B_zM - R[5]*R[7] ) / ( SR - SM )

    U_LM[0] = L[0]*( SL-uL ) / ( SL - SM )
    U_LM[1] = U_LM[0]*SM
    U_LM[2] = L[0]*vL*( SL - uL ) / ( SL - SM ) - ( B_xM*B_yM - L[5]*L[6] ) / ( SL - SM )
    U_LM[3] = L[0]*wL*( SL - uL ) / ( SL - SM ) - ( B_xM*B_zM - L[5]*L[7] ) / ( SL - SM )

    u_RM = [ U_HLL[1]/U_HLL[0], U_HLL[2]/U_HLL[0], U_HLL[3]/U_HLL[0]]
    u_R = [uR, vR, wR]
    u_LM = [ U_HLL[1]/U_HLL[0], U_HLL[2]/U_HLL[0], U_HLL[3]/U_HLL[0]]
    u_L = [uL, vL, wL]

    U_RM[4] = R[4]*( SR-uR ) / ( SR - SM ) + (  P_RM*SM - P_R*uR  - \
             B_xM*np.dot( B_RM, u_RM ) - R[5]*np.dot( B_R, u_R) )  / ( SR - SM )
    U_RM[5] = B_xM
    U_RM[6] = B_yM
    U_RM[7] = B_zM

    U_LM[4] = L[4]*( SL-uL ) / ( SL - SM) + (  P_LM*SM - P_L*uL  - \
             B_xM*np.dot( B_LM, u_LM ) - L[5]*np.dot( B_L, u_L) )  / ( SL - SM )
    U_LM[5] = B_xM
    U_LM[6] = B_yM
    U_LM[7] = B_zM

#  compute the fluxes of the left and right states

    flux_R_M = flux_R + SR*(U_RM - R)
    flux_L_M = flux_L + SL*(U_LM - L)

    if SL >= 0:
        flux = flux_L.copy()
    elif SM >= SL:
        flux = flux_L_M.copy()
    elif SM <= SR:
        flux = flux_R_M.copy()
    else:
        flux = flux_R.copy()

    return flux

# -------------------------------------------------------------------
# initialize animation
# -------------------------------------------------------------------
def init():
    line_d.set_xdata( x )
    line_u.set_xdata( x )
    line_v.set_xdata( x )
    line_by.set_xdata( x )
    line_p.set_xdata( x )
    return line_d, line_u, line_v, line_by, line_p

def update( frame ):
    #  plot
    d  = U[nghost:N-nghost,0]
    u  = U[nghost:N-nghost,1] / U[nghost:N-nghost,0]
    v  = U[nghost:N-nghost,2] / U[nghost:N-nghost,0]
    by = U[nghost:N-nghost,6]
    P  = ComputePressure( U[nghost:N-nghost,0], U[nghost:N-nghost,1], U[nghost:N-nghost,2], U[nghost:N-nghost,3],
                          U[nghost:N-nghost,4], U[nghost:N-nghost,5], U[nghost:N-nghost,6], U[nghost:N-nghost,7] )
    line_d.set_ydata( d )
    line_u.set_ydata( u )
    line_v.set_ydata( v )
    line_by.set_ydata( by )
    line_p.set_ydata( P )
#  ax[0].legend( loc='upper right', fontsize=12 )
#  ax[1].legend( loc='upper right', fontsize=12 )
#  ax[2].legend( loc='upper right', fontsize=12 )
    ax[0].set_title( 't = %6.3f' % (t) )

    return line_d, line_u, line_v, line_by, line_p


def update( frame ):
    global t, U

#  for frame==0, just plot the initial condition
    if frame > 0:
        for step in range( nstep_per_image ):

#        set the boundary conditions
            BoundaryCondition( U )

#        estimate time-step from the CFL condition
            dt = ComputeTimestep( U )
            print( "t = %13.7e --> %13.7e, dt = %13.7e" % (t,t+dt,dt) )

#        data reconstruction
            L, R = DataReconstruction_PLM( U )

#        update the face-centered variables by 0.5*dt
            for j in range( 1, N-1 ):
                flux_L = Conserved2Flux( L[j] )
                flux_R = Conserved2Flux( R[j] )
                dflux  = 0.5*dt/dx*( flux_R - flux_L )
                L[j]  -= dflux
                R[j]  -= dflux

#        compute fluxes
            flux = np.empty( (N,8) )
            for j in range( nghost, N-nghost+1 ):
#           R[j-1] is the LEFT state at the j+1/2 inteface
                flux[j] = HLLC( R[j-1], L[j]  )

#        update the volume-averaged input variables by dt
            U[nghost:N-nghost] -= dt/dx*( flux[nghost+1:N-nghost+1] - flux[nghost:N-nghost] )

#        update time
            t = t + dt
            if ( t >= end_time ):
                anim.event_source.stop()
                break
    #  plot
    d  = U[nghost:N-nghost,0]
    u  = U[nghost:N-nghost,1] / U[nghost:N-nghost,0]
    v  = U[nghost:N-nghost,2] / U[nghost:N-nghost,0]
    by = U[nghost:N-nghost,6]
    P  = ComputePressure( U[nghost:N-nghost,0], U[nghost:N-nghost,1], U[nghost:N-nghost,2], U[nghost:N-nghost,3],
                          U[nghost:N-nghost,4], U[nghost:N-nghost,5], U[nghost:N-nghost,6], U[nghost:N-nghost,7] )
    line_d.set_ydata( d )
    line_u.set_ydata( u )
    line_v.set_ydata( v )
    line_by.set_ydata( by )
    line_p.set_ydata( P )
#  ax[0].legend( loc='upper right', fontsize=12 )
#  ax[1].legend( loc='upper right', fontsize=12 )
#  ax[2].legend( loc='upper right', fontsize=12 )
    ax[0].set_title( 't = %6.3f' % (t) )

    return line_d, line_u, line_v, line_by, line_p

#--------------------------------------------------------------------
# parameters
#--------------------------------------------------------------------
# constants
L        = 1.0       # 1-D computational domain size
N_In     = 400       # number of computing cells
cfl      = 0.475       # Courant factor
nghost   = 2         # number of ghost zones
gamma    = 5.0/3.0   # ratio of specific heats
end_time = 0.08      # simulation time

# derived constants
N  = N_In + 2*nghost    # total number of cells including ghost zones
dx = L/N_In             # spatial resolution

# plotting parameters
nstep_per_image = 1     # plotting frequency


#--------------------------------------------------------------------
# main
#--------------------------------------------------------------------
# set initial condition
t = 0.0
x = np.empty( N_In )
U = np.empty( (N,8 ) )
for j in range( N_In ):
    x[j] = (j+0.5)*dx    # cell-centered coordinates
    U[j+nghost] = InitialCondition( x[j] )

BoundaryCondition( U )

# create figure
fig, ax = plt.subplots( 5, 1, sharex=True, sharey=False, figsize=(8,12) )
fig.subplots_adjust( hspace=0.15, wspace=0.0 )
#fig.set_size_inches( 3.3, 12.8 )
line_d,  = ax[0].plot( [], [], 'r-o', markeredgecolor='k', markersize=3 )
line_u,  = ax[1].plot( [], [], 'b-o', markeredgecolor='k', markersize=3 )
line_v,  = ax[2].plot( [], [], 'g-o', markeredgecolor='k', markersize=3 )
line_by, = ax[3].plot( [], [], 'g-o', markeredgecolor='k', markersize=3 )
line_p,  = ax[4].plot( [], [], 'k-o', markeredgecolor='k', markersize=3 )
ax[2].set_xlabel( 'x' )
ax[0].set_ylabel( 'Density' )
ax[1].set_ylabel( r'$v_x$' )
ax[2].set_ylabel( r'$v_y$' )
ax[3].set_ylabel( r'$B_y$' )
ax[4].set_ylabel( 'Pressure' )
ax[0].set_xlim( 0.30, L-0.25 )
ax[0].set_ylim( +0.0, 1.2 )
ax[1].set_ylim( -1.0, 1.0 )
ax[2].set_ylim( -2.0, 2.0 )
ax[3].set_ylim( -1.5, 1.5 )
ax[4].set_ylim( 0.0, 2.5)

# create movie
nframe = 99999999 # arbitrarily large
anim   = animation.FuncAnimation( fig, func=update, init_func=init,
                                  frames=nframe, interval=200, repeat=False )
plt.show()
