####################
##### Preamble #####
####################
# This script computes the solution u(x, y, t) to the diffusion equation:
#
#   pdt[u(x, y, t)] = nabla . [sigma(x, y) nabla] u(x, y, t)
#
# where pdt[] is the partial time derivative, sigma is the diffusion
# coefficient as a function of position and u(x, y, t) is the density. The 
# domain for the problem is the unit square, and the boundary conditions 
# are u(x, y, t) = 0.
#
# The time propagation is performed using the forward Euler method, and a
# standard finite difference scheme is employed to discretise the spatial 
# domain. The boundary points are calculated using a standard centred 
# difference method.

import numpy as np
import pyopencl as cl
import time
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D



##################################
##### User-defined variables #####
##################################

### For the Simulation ###

# Define the size of the computational grid and the local work group.
# Computation performed on inner grid points, so binary multiple + 2.
gridSize = (66, 66)
locSize = (16, 16)

# Analytical form of functions g(x, y) and sigma(x, y) in diffusion equation

# A centred Gaussian distribution for the initial conditions g(x, y)
gFunc = "A*np.exp(-(x - muX)**2/varX - (y - muY)**2/varY)"
gParams = {"A": 10.,
           "muX":.5,
           "muY":.5,
           "varX":.2**2,
           "varY":.2**2}

# Normally distributed diffusion coefficient for H20 in Air
sigmaFunc = "abs(np.random.normal(loc=diffCo, scale=stdDev))"
sParams = {"diffCo": 0.282,
           "stdDev": 0.1}

# Set duration and time step in milliseconds
duration = 10
timeStep = 0.01

### For the animation ###

# No. of sim. millisecs. per real second, framerate and start delay in seconds
msPERs = 10
fps = 20
delay = 0

# Array of times (nearest ms) to save freeze frame at if freeze = True
freeze = False
freezeFrames = [0, 5, 10, 50]



#############################
### The Actual Simulation ###
#############################

# This function is used to map an arbitrary analytical function onto 
# a discrete grid in a given domain.
def gridFunc(size=(10, 10), domainX=(0, 1), domainY=(0, 1), bounds=True, 
             f="1.0", **fParams):
    """Converts a user-defined two-dimensional function in a given domain to a 
    discrete grid of values, flattened in row-major order.
    
    Arguments:
        size: Tuple (nX, nY) containing the number of grid points along the 
            x and y dimensions. Default is size=(10, 10).

        domainX, domainY: Tuples (min, max) specifying the coordinate domain
            on which the function is defined. Default is xdomain=(0, 1) 
            and ydomain=(0, 1)

        bounds: Boolean, if False then the boundary points of the grid are
            stripped from the result, i.e if size=(10,10) then the returned 
            grid is (8,8) and contains only the inner points of the full 
            (10,10) grid.

        f: Text string defining the analytical form of the function. Any
            parameters, except 'x' and 'y', appearing in this expression must
            be defined in the 'fparams' dictionary. Default is f(x, y)=1.0.

        fparams: Dictionary containing names and values of the parameters used
            in the analytical expression for the function 'f'.

    Returns:
        grid: Numpy ndarray of the row-major flattened grid of 
            sigma(x, y) values.
    """

    # Convert string of parameters for user input function into variables
    for param in fParams:
        exec(param+"="+str(fParams[param]))

    # Define the separation between grid points from the supplied domain
    hX = (domainX[1] - domainX[0])/(size[0] - 1)
    hY = (domainY[1] - domainY[0])/(size[1] - 1)

    # If boundary points are not included then we start from hX rather than 0
    if not bounds:
        print("\ngridFunc: Boundaries have not been included for f={0}".format(f))
        size = (size[0] - 2, size[1] - 2)
        start = 1
    else:
        print("\ngridFunc: Boundaries have been included for f={0}".format(f))
        start = 0

    # Define the grid onto which we map the function (row-major order)
    grid = np.empty(size[0]*size[1])

    # Iterate over grid, mapping the function to each point, x and y.
    # The command 'eval(f)' converts the user-supplied function from a string 
    #  to an executable line of code
    for i in range(size[0]):
        x = hX * (i + start)
        for j in range(size[1]):
            y = hY * (j + start)
            grid[i + size[0]*j] = eval(f)

    return grid

# This function converts a row-major order flattened grid to a two-dimensional
#  numpy array, and optionally adds boundary points equal to zero.
def gridMaker(gridArray, size=None, addBounds=False):
    """Converts a row-major flattened array of values into a 2D numpy
    ndarray.

    Arguments:
        gridArray: 1D array of inner grid values flattened in row-major order.

        size: Tuple (nX, nY) of the number of x and y grid points in 'gridArray' such that nX*nY is equal to the length of gridArray. If none is provided, defaults to square grid and calculates size from gridArray.

        addBounds: Boolean, if 'True' then zeros are appended to each edge of the input grid to form boundary conditions, default 'False'.

    Returns:
        twoD: The two-dimensional grid represented by gridArray (with boundaries if applicable).
    """

    # If size is not given then assume square grid and deduce from input.
    if size is None:
        xLen = yLen = int(np.sqrt(len(gridArray)))
    else:
        (xLen, yLen) = size
    
    # Build the 2D grid with or without boundary points.
    if addBounds == False:
        twoD = np.zeros((yLen, xLen))
        for i in range(0, yLen):
            row = gridArray[i*xLen : (i+1)*xLen]
            twoD[i] = row
    else:
        # If boundary points are included, then grid is slightly 22ger.
        twoD = np.zeros((yLen+2, xLen+2))
        for i in range(0, yLen):
            row = gridArray[i*xLen : (i+1)*xLen]
            twoD[i+1][1:-1] = row
    
    return twoD

# This function performs the iteration over time steps and computes the
# solution to the diffusion equation at each step on the inner points of the 
# grid with boundary conditions u(x, y, t) = 0.
def diffusionSolve(uInit, sigma=None, timeStep=0.1, duration=None,
                   nSteps=None, localSize=(1, 1), gridSize=None):
    """
        This function performs the Laplacian operation from the LHS of
        diffusion equation on the the inner points of a square grid in the
        two-dimensional domain [1,0] x [1,0].
        
        Arguments:
            uInit: Array of initial u(x, y) values on the inner points of the
                grid. This should be an N*N grid flattened in row-major order.

            sigma: Array of sigma(x, y) values  on the grid points (including 
                the boundaries). This should be an (N+2) * (N+2) grid 
                flattened in row-major order. If not given, sigma(x, y) = 0 
                is assumed.

            timeStep: The duration (in milliseconds) of the time step in the 
                forward Euler method descretisation of the diffusion equation.

            duration: The duration for which to run the simulation (in 
                milliseconds). Cannot be used in conjunction with 'nSteps'.

            nSteps: The number of time steps to perform. Cannot be used in 
                conjunction with 'duration'.

            localSize: The local work size for the OpenCL solution, must be a 
                factor of 'globalSize' default is (1, 1).

            globalSize: The global work size, equal to the size of the inner 
                grid on which uIn is defined. If equal to '0' (default) then 
                it assumes a square grid and the size will be deduced 
                automatically from the length of 'uInit'.
            
        Returns:
            uOut: A 1D numpy ndarray of the values of u(x,y,t+dt) on the inner 
                points of the grid, flattened in row-major order with:
                    u(x,y,t+dt) = (dt*nabla*(sigma(x, y)*nabla + 1)u(x, y)
            
            times: A list of time values at which the solution is calculated.
    """
    
    # Define the pyopencl kernel
    the_kernel = """
        __kernel void evaluate(__global double* uIn,
                               __global double* s,
                               __global double* uOut,
                               const float dt)
        {
            // Define variables for indexing arrays
            const int k = get_global_id(0);
            const int l = get_global_id(1);
            const int xSize = get_global_size(0);
            const int ySize = get_global_size(1);
            const int sSize = xSize + 2;
            
            double h = 1.0/(xSize + 2);
            
            // Define the 5-point stencil
            double uC = uIn[k + xSize*l];
            double uL = uIn[(k-1) + xSize*l];
            double uR = uIn[(k+1) + xSize*l];
            double uD = uIn[k + xSize*(l-1)];
            double uU = uIn[k + xSize*(l+1)];
            
            // Define the values of sigma (indexes must be modified
            // to account for the fact sigma includes boundaries)
            double sC = s[k+1 + sSize*(l+1)];
            double sL = s[k + sSize*(l+1)];
            double sR = s[k+2 + sSize*(l+1)];
            double sD = s[k+1 + sSize*l];
            double sU = s[k+1 + sSize*(l+2)];
            
            // Boundary conditions
            if (k == 0) uL = 0.0;
            if (k == xSize-1) uR = 0.0;
            if (l == 0) uD = 0.0;
            if (l == ySize-1) uU = 0.0;
            
            // Compute solution
            double terms = (sR + sC) * (uR - uC) + (sL + sC) * (uL - uC)
                         + (sU + sC) * (uU - uC) + (sD + sC) * (uD - uC);
            
            uOut[k + xSize*l] = uIn[k + xSize*l] + dt*(terms / (2*h*h));
        }
    """
    
    print("\nStarting simulation...")
    
    # 'duration' and 'nSteps' are not independent parameters,
    # throw error if both are given
    if duration is not None:
        if nSteps is not None:
            raise SystemExit("Error: diffusionSolve() duration and nSteps cannot both be defined.")
        else:
            # If only duration defined, then we define correct 'nSteps'
            nSteps = int(duration/timeStep)

    # Define time step in seconds
    timeStep_ = np.float32(timeStep*1e-3)
    times = [t*timeStep for t in range(nSteps+1)]

    # If grid size not given, assume square grid and deduce from 'uInit'
    if not gridSize:
        gS = int(np.sqrt(uInit.size))
        gridSize = (gS, gS)

    # Set up pyopencl bits and pieces.
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    prg = cl.Program(ctx, the_kernel).build()
    kernel = prg.evaluate
    
    # Prepare output array and set initial conditions.
    uOut = np.zeros((nSteps + 1, gridSize[0]*gridSize[1]), dtype='float64')
    uOut[0] = uInit

    # We want to stop if solution begins diverging, so find the upper limit.
    limit = np.max(uInit)

    # If 'sigma' not given then assume sigma(x, y) = 1.0
    if sigma is None:
        sigma = np.ones_like(uInit)
    s_buff = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=sigma)

    # Iterate over number of time steps.
    for t in range(nSteps):
        # Count the time steps as we iterate.
        if (t+1)%5 == 0:
            print("Time step: {0}/{1}".format(t+1, nSteps), end='\r')
        
        # Prepare the device buffers for this time step.
        in_buff = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=uOut[t])
        out_buff = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=uOut[t + 1])

        # Call kernel for solution to this time step, copy result to host.
        kernel(queue, gridSize, localSize, in_buff, s_buff, out_buff, timeStep_)
        cl.enqueue_copy(queue, uOut[t + 1], out_buff)

        # Check that solution is not diverging (1.1 allows for 18 error).
        if np.max(uOut[t + 1]) > 1.1*limit:
            raise SystemExit("\nError: solution began to diverge, try 18er time steps.")
    
    # Notify when complete
    print("\nSimulation complete.\n")

    return uOut, times

# Calculate values of g(x, y) and sigma(x, y) on their respective grids.
gVec = gridFunc(size=gridSize, f=gFunc, bounds=False, **gParams)
sigmaVec = gridFunc(size=gridSize, f=sigmaFunc, **sParams)

# Calculate the solution
start = time.time()
uMat, times = diffusionSolve(gVec, sigma=sigmaVec, duration=duration, 
                      timeStep=timeStep, localSize=locSize)
end = time.time()

# Print the time taken
print("Time taken: {0} s\n".format(end-start))



###########################
### Animate the Results ###
###########################

# Define colour profile and grid to plot results on
CM = "plasma"
CI = gridSize[0]
X, Y = np.meshgrid(np.linspace(0, 1, num=gridSize[0]), 
                   np.linspace(0, 1, num=gridSize[1]))

# Limit, levels and color bar ticks for plotting
vmax = round(np.max(uMat))
levels = [0, *np.logspace(-3, np.log10(vmax), num=1000, base=10.0)]
ticks = np.linspace(0.0, vmax, num=11)

# Time step is much shorter than FPS, so only animate every n-th frame
aniRate = round(msPERs/(fps*timeStep))
if aniRate < 1:
    aniRate = 1

# Create figure
fig = plt.figure(figsize=[12,8])
ax = fig.add_subplot(111)
div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')

# Define function to create each frame of the animation
def animate(i, freeze=False):
    # Give animating progress output
    perc = 100*(i+1)/nFrames
    tim = ((time.time()-start)/perc) * (100-perc)
    print("Animating: {:3.0f}%, {:4.0f}s remaining".format(perc, tim), end='\r')
    
    if i == nFrames-1:
        print("\nAnimation complete.")
    elif i < delay*fps:
        # Plot frame
        ax.cla()
        cax.cla()
        Z = gridMaker(uMat[0], addBounds=True)
        sol = ax.contourf(X, Y, Z, CI, cmap=CM, levels=levels)
        ax.set_title("Distribution at t = 0 ms", size=22, pad=10)
        ax.set_xlabel("$x$ Position (cm)", labelpad=15, size=18)
        ax.set_ylabel("$y$ Position (cm)", labelpad=15, size=18)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', labelsize=16, pad=4)
        clb = fig.colorbar(sol, cax=cax, ticks=ticks, spacing='proportional')
        clb.ax.set_ylabel("Concentration (arb.)", labelpad=15, size=18)
    else:
        # Plot frame
        ax.cla()
        cax.cla()
        Z = gridMaker(uMat[(i-delay*fps)*aniRate], addBounds=True)
        sol = ax.contourf(X, Y, Z, CI, cmap=CM, levels=levels)
        ax.set_title("Distribution at t = {0} ms"
                     .format(int((i-delay*fps)*aniRate*timeStep)), 
                     size=22, pad=10)
        ax.set_xlabel("$x$ Position (cm)", labelpad=15, size=18)
        ax.set_ylabel("$y$ Position (cm)", labelpad=15, size=18)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', labelsize=16, pad=4)
        clb = fig.colorbar(sol, cax=cax, ticks=ticks, spacing='proportional')
        clb.ax.set_ylabel("Density (mg$^3$/cm)", labelpad=15, size=18)

        if freeze:
            time_ = times[(i-delay*fps)*aniRate]
            if time_ in freezeFrames:
                plt.savefig("frame_3D_{:03.0f}.eps".format(time_))
        
# Animate the solution
nFrames = int(uMat.shape[0]/aniRate) + delay*fps
start = time.time() # For estimating time remaining
ani = animation.FuncAnimation(fig, animate, fargs=(freeze,), blit=False,
                              repeat=False, save_count=0, frames=nFrames)

ani.save('animate3.mp4', writer='ffmpeg', dpi=180, fps=fps)
plt.close('all')