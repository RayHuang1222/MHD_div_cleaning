#--------------------------------------------------------------------
# Test the Sod's shock tube with MPI.
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
N_In     = 128       # number of computing cells
n_In     = int(N_In/nrank)# number of computing cells per process
cfl      = 1.0       # Courant factor
nghost   = 2         # number of ghost zones
gamma    = 5.0/3.0   # ratio of specific heats
end_time = 0.1       # simulation time

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
#  Sod shock tube
   if ( x < 0.5*L ):
      d = 1.0  # density
      u = 0.0  # velocity x
      v = 0.0
      w = 0.0
      P = 1.0  # pressure
      E = P/(gamma-1.0) + 0.5*d*u**2.0    # energy density
   else:
      d = 1.25e-1
      u = 0.0
      v = 0.0
      w = 0.0
      P = 0.1
      E = P/(gamma-1.0) + 0.5*d*u**2.0

#  conserved variables [0/1/2/3/4] <--> [density/momentum x/momentum y/momentum z/energy]
   return np.array( [d, d*u, d*v, d*w, E] )


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


# -------------------------------------------------------------------
# compute pressure
# -------------------------------------------------------------------
def ComputePressure( d, px, py, pz, e ):
   P = (gamma-1.0)*( e - 0.5*(px**2.0 + py**2.0 + pz**2.0)/d )
   assert np.all( P > 0 ), "negative pressure !!"
   return P


# -------------------------------------------------------------------
# compute time-step by the CFL condition
# -------------------------------------------------------------------
def ComputeTimestep( U ):
   P = ComputePressure( U[:,0], U[:,1], U[:,2], U[:,3], U[:,4] )
   a = ( gamma*P/U[:,0] )**0.5
   u = np.abs( U[:,1]/U[:,0] )
   v = np.abs( U[:,2]/U[:,0] )
   w = np.abs( U[:,3]/U[:,0] )

#  maximum information speed in 3D
   max_info_speed = np.amax( u + a )
   dt_cfl         = cfl*dx/max_info_speed

   return dt_cfl


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

   return slope_limited


# -------------------------------------------------------------------
# convert conserved variables to primitive variables
# -------------------------------------------------------------------
def Conserved2Primitive( U ):
   W = np.empty( 5 )

   W[0] = U[0]
   W[1] = U[1]/U[0]
   W[2] = U[2]/U[0]
   W[3] = U[3]/U[0]
   W[4] = ComputePressure( U[0], U[1], U[2], U[3], U[4] )

   return W


# -------------------------------------------------------------------
# convert primitive variables to conserved variables
# -------------------------------------------------------------------
def Primitive2Conserved( W ):
   U = np.empty( 5 )

   U[0] = W[0]
   U[1] = W[0]*W[1]
   U[2] = W[0]*W[2]
   U[3] = W[0]*W[3]
   U[4] = W[4]/(gamma-1.0) + 0.5*W[0]*( W[1]**2.0 + W[2]**2.0 + W[3]**2.0 )

   return U


# -------------------------------------------------------------------
# piecewise-linear data reconstruction
# -------------------------------------------------------------------
def DataReconstruction_PLM( U ):

#  allocate memory
   W = np.empty( (n,5) )
   L = np.empty( (n,5) )
   R = np.empty( (n,5) )

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
   flux = np.empty( 5 )

   P = ComputePressure( U[0], U[1], U[2], U[3], U[4] )
   u = U[1] / U[0]

   flux[0] = U[1]
   flux[1] = u*U[1] + P
   flux[2] = u*U[2]
   flux[3] = u*U[3]
   flux[4] = u*( U[4] + P )

   return flux


# -------------------------------------------------------------------
# Roe's Riemann solver
# -------------------------------------------------------------------
def Roe( L, R ):
#  compute the enthalpy of the left and right states: H = (E+P)/rho
   P_L = ComputePressure( L[0], L[1], L[2], L[3], L[4] )
   P_R = ComputePressure( R[0], R[1], R[2], R[3], R[4] )
   H_L = ( L[4] + P_L )/L[0]
   H_R = ( R[4] + P_R )/R[0]

#  compute Roe average values
   rhoL_sqrt = L[0]**0.5
   rhoR_sqrt = R[0]**0.5

   u  = ( L[1]/rhoL_sqrt + R[1]/rhoR_sqrt ) / ( rhoL_sqrt + rhoR_sqrt )
   v  = ( L[2]/rhoL_sqrt + R[2]/rhoR_sqrt ) / ( rhoL_sqrt + rhoR_sqrt )
   w  = ( L[3]/rhoL_sqrt + R[3]/rhoR_sqrt ) / ( rhoL_sqrt + rhoR_sqrt )
   H  = ( rhoL_sqrt*H_L  + rhoR_sqrt*H_R  ) / ( rhoL_sqrt + rhoR_sqrt )
   V2 = u*u + v*v + w*w
#  check negative pressure
   assert H-0.5*V2 > 0.0, "negative pressure!"
   a  = ( (gamma-1.0)*(H - 0.5*V2) )**0.5 

#  compute the amplitudes of different characteristic waves
   dU     = R - L
   amp    = np.empty( 5 )
   amp[2] = dU[2] - v*dU[0]
   amp[3] = dU[3] - w*dU[0]
   amp[1] = (gamma-1.0)/a**2.0 \
            *( dU[0]*(H-u**2.0) + u*dU[1] - dU[4] + v*amp[2] + w*amp[3] )
   amp[0] = 0.5/a*( dU[0]*(u+a) - dU[1] - a*amp[1] )
   amp[4] = dU[0] - amp[0] - amp[1]

#  compute the eigenvalues and right eigenvector matrix
   EigenValue    = np.array( [u-a, u, u, u, u+a] )
   EigenVector_R = np.array(  [ [1.0, u-a,   v,   w,  H-u*a],
                                [1.0,   u,   v,   w, 0.5*V2],
                                [0.0, 0.0, 1.0, 0.0,      v],
                                [0.0, 0.0, 0.0, 1.0,      w],
                                [1.0, u+a,   v,   w,  H+u*a] ]  )

#  compute the fluxes of the left and right states
   flux_L = Conserved2Flux( L )
   flux_R = Conserved2Flux( R )

#  compute the Roe flux
   amp *= np.abs( EigenValue )
   flux = 0.5*( flux_L + flux_R ) - 0.5*amp.dot( EigenVector_R )

   return flux

# -------------------------------------------------------------------
# HLL Scheme
# -------------------------------------------------------------------
def HLL( L, R ):
    
    rhoL_sqrt = L[0]**0.5
    rhoR_sqrt = R[0]**0.5

#  compute the enthalpy of the left and right states: H = (E+P)/rho
    P_L = ComputePressure( L[0], L[1], L[2], L[3], L[4] )
    P_R = ComputePressure( R[0], R[1], R[2], R[3], R[4] )
    H_L = ( L[4] + P_L )/L[0]
    H_R = ( R[4] + P_R )/R[0]
    
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
    bL = min( u-a, L[1]/(rhoL_sqrt)**2-cL )
    bR = max( u+a, R[1]/(rhoR_sqrt)**2+cR )

    SR = max( bR, 0 )
    SL = min( bL, 0 )

#  compute the fluxes of the left and right states
    flux_L = Conserved2Flux( L )
    flux_R = Conserved2Flux( R )

#  compute the HLL flux
    flux_HLL = ( SR*flux_L - SL*flux_R + SR*SL*( R-L ) ) / ( SR-SL )

    if SL >= 0:
        flux = flux_L.copy()
    elif SL < 0 and SR > 0 :
        flux = flux_HLL.copy()
    else:
        flux - flux_R.copy()
    
    return flux

# -------------------------------------------------------------------
# HLLC Scheme
# -------------------------------------------------------------------
def HLLC( L, R ):
    
    rhoL_sqrt = L[0]**0.5
    rhoR_sqrt = R[0]**0.5

#  compute the enthalpy of the left and right states: H = (E+P)/rho
    P_L = ComputePressure( L[0], L[1], L[2], L[3], L[4] )
    P_R = ComputePressure( R[0], R[1], R[2], R[3], R[4] )
    H_L = ( L[4] + P_L )/L[0]
    H_R = ( R[4] + P_R )/R[0]
    
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
    bL = min( u-a, L[1]/(rhoL_sqrt)**2-cL )
    bR = max( u+a, R[1]/(rhoR_sqrt)**2+cR )

    SR = max( bR, 0 )
    SL = min( bL, 0 )
    
#  compute the interface speed    

    uR = R[1]/(rhoR_sqrt)**2
    uL = L[1]/(rhoL_sqrt)**2
    
    SM = ( P_R - P_L + L[0]*uL*( SL - uL ) - R[0]*uR*( SR - uR ) ) / ( L[0]*( SL - uL ) - R[0]*( SR - uR ))

    DM = np.array( [0.0, 1.0, 0.0 ,0.0, SM] )
    
#  compute the fluxes of the left and right states
    flux_L = Conserved2Flux( L )
    flux_R = Conserved2Flux( R )
    
    flux_R_M = ( SM*( SR*R - flux_R ) + SR*( P_R + R[0]*( SR - uR )*( SM - uR ) )*DM ) / ( SR - SM )
    flux_L_M = ( SM*( SL*L - flux_L ) + SL*( P_L + L[0]*( SL - uL )*( SM - uL ) )*DM ) / ( SL - SM )
    
    if SL >= 0:
        flux = flux_L.copy()
    elif SL < 0 and SM >= 0:
        flux = flux_L_M.copy()
    elif SM < 0 and SR >= 0:
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
   line_p.set_xdata( x )
   return line_d, line_u, line_p

# -------------------------------------------------------------------
# update animation
# -------------------------------------------------------------------
def update( frame ):
   global t, U
   
   # split the cells into two halfs
   u = np.empty((n,5))
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
         dt_cfl = ComputeTimestep( u )
         tsend = np.array([dt_cfl])
         trecv = np.zeros(1)
         if rank == 0:
            comm.Send(tsend, des, tag = it+1000)
            comm.Recv(trecv, src, tag = it+1001)
         else:
            comm.Recv(trecv, src, tag = it+1000)
            comm.Send(tsend, des, tag = it+1001)
         dt_cfl2 = trecv[0]
         dt_end = end_time-t
         dt = min( dt_cfl, dt_cfl2, dt_end )
         
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
         flux = np.empty( (n,5) )
         for j in range( nghost, n-nghost+1 ):
#           R[j-1] is the LEFT state at the j+1/2 inteface
            flux[j] = HLLC( R[j-1], L[j]  )

#        update the volume-averaged input variables by dt
         u[nghost:n-nghost] -= dt/dx*( flux[nghost+1:n-nghost+1] - flux[nghost:n-nghost] )

#        send boundary data
         recv = np.empty((nghost,5))
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

#        update time
         t = t + dt
         if ( t >= end_time ):
            anim.event_source.stop()
            break

#  combine data
   Recv = np.empty((n_In,5))
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
   d = U[nghost:N-nghost,0]
   u = U[nghost:N-nghost,1] / U[nghost:N-nghost,0]
   P = ComputePressure( U[nghost:N-nghost,0], U[nghost:N-nghost,1], U[nghost:N-nghost,2],
                        U[nghost:N-nghost,3], U[nghost:N-nghost,4] )
   line_d.set_ydata( d )
   line_u.set_ydata( u )
   line_p.set_ydata( P )
   ax[0].set_title( 't = %6.3f, rank = %d' % (t, rank) )

   return line_d, line_u, line_p


#--------------------------------------------------------------------
# main
#--------------------------------------------------------------------

# set initial condition
t = 0.0
x = np.empty( N_In )
U = np.empty( (N,5) )
for j in range( N_In ):
   x[j] = (j+0.5)*dx    # cell-centered coordinates
   U[j+nghost] = InitialCondition( x[j] )

# create figure
fig, ax = plt.subplots( 3, 1, sharex=True, sharey=False, dpi=140 )
fig.subplots_adjust( hspace=0.1, wspace=0.0 )
line_d, = ax[0].plot( [], [], 'r-o', markeredgecolor='k', markersize=3 )
line_u, = ax[1].plot( [], [], 'b-o', markeredgecolor='k', markersize=3 )
line_p, = ax[2].plot( [], [], 'g-o', markeredgecolor='k', markersize=3 )
ax[2].set_xlabel( 'x' )
ax[0].set_ylabel( 'Density' )
ax[1].set_ylabel( 'Velocity' )
ax[2].set_ylabel( 'Pressure' )
ax[0].set_xlim( 0.30, 1.0-0.25 )
ax[0].set_ylim( +0.0, 1.25 )
ax[1].set_ylim( -0.2, 1.0 )
ax[2].set_ylim( +0.0, 1.5 )

# create movie
nframe = 99999999 # arbitrarily large
anim   = animation.FuncAnimation( fig, func=update, init_func=init,
                                  frames=nframe, interval=200, repeat=False )
plt.show()
