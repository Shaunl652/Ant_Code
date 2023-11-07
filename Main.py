# -*- coding: utf-8 -*-

# =============================================================================
# Code designed to find the average time it takes for an ant starting as (0,0) to
# reach food on a line
# Uses a 2D probability distribution at various time steps to estimate the likelihood
# of the ant finnding foot at a time t, then uses this function to find the probability
# that the ant will reach the final point at a time t
# This final function can be approximated by a Lorentzian(?) which we can curve fit
# to find the mean 
# =============================================================================


import numpy as np
import matplotlib.pyplot as plt


def not_edge(x,y,xlen,ylen):
    # returns False if the location (x,y) is on the edge of the grid
    out=True
    if x==0:
        out = False
    if x==xlen: 
        out = False
    if y==0:
        out=False
    if y==ylen:
        out = False

    return out

def not_food(x,y,Locs):
    # Checks if the location x,y is a food site
    # Returns False if x,y IS a food site
    out = True
    for l in Locs:
        if x == l[0] and y == l[1]:
            out = False
            
    return out
    
def grid_sum(grid,x,y,xlen,ylen,food_locs):
    # Sums the probabilities coming from each adjacent site
    # We have the /4 as the probability at each site goes 4 ways
    
    # the ant can only move to a point that has a 0 probability of being occupied
    # unless that site is a food site
    if not_food(x, y, food_locs):
        total = 0 # initilise the variable
    else:
        total = grid[x,y]
    # First we check if the point x,y is on an edge otherwise looking to the side will cause issues
    # We also check that the point we are looking at isn't a food site as the 
    # ant should't move from a food site
    if x==0 :
        if not_food(x+1,y,food_locs):
            total += grid[x+1,y]/4
    elif x==xlen:
        if not_food(x-1,y,food_locs): 
            total += grid[x-1,y]/4
    elif y==0 :
        if not_food(x,y+1,food_locs):
            total += grid[x,y+1]/4
    elif y==ylen:
        if not_food(x,y-1,food_locs):
            total += grid[x,y-1]/4
    else:
        # We want to check that the site we're looking at coming from is not a food site
        # Because the ant will NOT leave a food site
        if not_food(x,y+1,food_locs):
            # If the above is true then we're NOT on an edge
            total += grid[x,y+1]/4
    
        if not_food(x+1,y,food_locs):
            # If the above is true then we're NOT on an edge
            total += grid[x+1,y]/4
    
        if not_food(x,y-1,food_locs):
            # If the above is true then we're NOT on an edge
            total += grid[x,y-1]/4
    
        if not_food(x-1,y,food_locs):
            # If the above is true then we're NOT on an edge
            total += grid[x-1,y]/4
    
    return total
        
    
def sum_food(grid,food_locs):
    # Sums the probability for each edge
    # This is the probability that the ant has found food
    output = 0
    for loc in food_locs:
        output+=grid[loc[0],loc[1]]
    return output

# This was coded to work for problem 1) then wrapped into a function to make it more generic 
def ant_func(axis_lims,food_locs,speed = 10,t_step = 1,N = 20,Plot=False):
    """
    Function to find the modal time it takes an ant on a random walk to reach food
    located at the points food_locs

    Parameters
    ----------
    axis_lims : FLOAT
        Grid sites in the x axis.
    food_locs : ARRAY OF TUPLE
        Grid sites where food is located.
    speed : FLOAT
        Speed of the ant in cm/s, default is 10.
    t_step : FLOAT
        length of time between each step in seconds, Default is 1.
    N : INT
        Number of time steps to simulate, Default is 20.
    Plot : Bool
        Decides if we want to plot the graphs

    Returns
    -------
    mode_time : Float
        modal time it takes for the ant to reach food.

    """
    # Number of time steps
    times = np.array([*range(N+1)])*t_step

    step_size = speed*t_step # distance traveled in 1 time step in cm

    # Returns the axis, as the problem is square use this for x and y, 21 ensures we get the +20 site
    axis = np.arange(-axis_lims,+axis_lims,step_size)
    ax_len = len(axis)
    mid = np.where(axis==0)[0][0]  


    # Initilise the var that stores the probability that the ant has found the food at each time step
    prob_food = [0] # at the start the ant is definitly not at the food
    
    
    # Set up the grid for t=0
    grid0 = np.zeros((ax_len+1,ax_len+1))
    grid0[mid,mid] = 1 # Start with 100% certainty that the ant is in the centre
    gridvals = [grid0]
    # Start looping over each of the time steps
    for t in range(1,N+1):
        # Grid after current time step
        grid1 = grid0.copy() # The copy method prevents changes to grid0 when changing grid1
    
        # Next loop through each of the grid sites on the grid at the start of the time step
        for y in range(ax_len+1):
            for x in range(ax_len+1):
                grid1[x,y]  = grid_sum(grid0,x,y,ax_len,ax_len,food_locs)
    
        grid0 = grid1.copy() # Set grid0 to the grid in the previous step
        gridvals.append(grid0)
        # Next we update the prob that the ant has found the food
        prob_food.append(sum_food(grid1,food_locs))
        
    
    # The probability of finding the food in time t is 2*(dP/dt)*P(t)
    # This equation is the skew norm: https://en.wikipedia.org/wiki/Skew_normal_distribution
    prob_time = 2*np.gradient(prob_food,times)  *np.array(prob_food)  

    # from scipy.stats import skewnorm
    
    # vals = skewnorm.fit(prob_time)
    # times_dense = np.linspace(0,N*t_step,1001)    
    
    mode_index = np.argmax(prob_time)
    mode_time = times[mode_index]
    if Plot:

        import matplotlib.pyplot as plt
    
        fig1, ax1 = plt.subplots()
        ax1.plot(times,prob_food,marker='x')
        ax1.set(title='Probability of the ant having found food after t seconds',xlabel='time [s]',ylabel='probability')
        
        fig2,ax2= plt.subplots()
        #ax2.plot(times_dense,skewnorm.pdf(times_dense,vals[0],vals[1],vals[2]))
        ax2.plot(times,prob_time,marker='x')
        ax2.set(title='Probability of the ant reaing food in t seconds',xlabel='time [s]',ylabel='probability')
        
    return mode_time,gridvals

# First we set up the problem


# =============================================================================
# First we do probelm 1)
# Food is located on lines 20 from the origin in each direction
# =============================================================================
speed = 10
t_step = 1
N = 20
# Number of time steps
times = np.array([*range(N+1)])*t_step

step_size = speed*t_step # distance traveled in 1 time step in cm
# Returns the axis, as the problem is square use this for x and y, 21 ensures we get the +20 site
axis = np.arange(-20,+21,step_size)
ax_len = len(axis)-1
mid = np.where(axis==0)[0][0]

# Defines the locations were the food is
food1 = [] # Initilise
# Add the top edge
for i in range(ax_len+1):
    food1.append((0,i))
# Add the left edge
for i in range(1,ax_len+1):
    food1.append((i,0))

# Add the bottom edge
for i in range(1,ax_len):
    food1.append((ax_len,i))
# Add the right edge
for i in range(1,ax_len+1):
    food1.append((i,ax_len))
ans1,grid1 = ant_func(20, food1,speed=speed,t_step=t_step,N=N)

# =============================================================================
# Now we can difine problem 2
# =============================================================================

# Find the locations of the food
def food_line(ax_lims):
    # We assume that we are using the default spacing and a square grid
    # This will simplify everythong and allow us to have an arbirty grid size
    
    # Define the line as a function of array indicies
    # Found the equaiton analytically via simulatenous equations
    line = lambda x: x-1

    # We want to keep the default time steps and grid spacing
    # This allows us to use the following axes (square grid)
    axes = np.arange(-ax_lims,+ax_lims,10)
    locs = []
    
    for x in range(1,len(axes)+1):
        locs.append((x,line(x)))
        
    return locs

food2 = food_line(50)

ans2,grid2 = ant_func(50, food2)

# import matplotlib.pyplot as plt



# =============================================================================
# Now we set up the final problem
# =============================================================================


# First we want a function that will make the food locations from a given function

# This is the functiondefining the area where the food is
# It should be cacluated the edges of this boundry before hand and this should inform 
# the edges of the grid
def circle(x,y):
    # Checks if the given x,y coord is inside or outside the defined area
    
    func = ((x-2.5)/30)**2 + ((y-2.5)/40)**2
    if func < 1:
        output = False
    else:
        output = True
    return output

def find_food_locs(f,axes):
    locs = []
    for xi,x in enumerate(axes):
        for yi,y in enumerate(axes):
            if f(x,y):
                locs.append((xi,yi))
    return locs
    
axes3 = np.arange(-60,61,step_size)
food3 = find_food_locs(circle,axes3)

if False:
    # Will plot the food sites if True
    xs_scatter = []
    ys_scatter = []
    for xy in food3:
        xs_scatter.append(xy[0])
        ys_scatter.append(xy[1])
    
    fig_s,ax_s = plt.subplots()
    
    ax_s.scatter(xs_scatter,ys_scatter)
    ax_s.set(title='Food site locations',xlabel='x [cm]',ylabel='y [cm]')


ans3,grid3 = ant_func(60, food3,N=50)

# Now we print the answers
print(f'Mode time for problem 1 = {ans1} s')
print(f'Mode time for problem 2 = {ans2} s')
print(f'Mode time for problem 3 = {ans3} s')


# =============================================================================
# Commented below is a block of code that will plot the likelihood that the ant will
# be at any given site at time step t
# To see the animation simply change the vars passed to anigrid, anifood, 
# and set the appropriate axlims
# This is currently set to plot problem 1)
# This will plot the food locations with red 'X's
# =============================================================================


# anigrid = grid1
# anifood = food1
# axlim = 20
# from matplotlib.animation import FuncAnimation
# figa,axa=plt.subplots()
# m=axa.pcolormesh(axis,axis,anigrid[0],vmax=1,vmin=0)
# figa.colorbar(m,label='Probability')
# axis = np.arange(-axlim,axlim+1,step_size)
# def animate(i):
#     axa.clear()
#     m=axa.pcolormesh(axis,axis,anigrid[i],vmax=1,vmin=0)
#     axa.set(title=f'time step = {times[i]}')
#     for f in anifood:
#         x = axis[f[0]]
#         y = axis[f[1]]
#         plt.text(y,x,'X',ha='center',va='center',color='r')
    
# ani = FuncAnimation(figa,animate,frames=N,interval=500)
