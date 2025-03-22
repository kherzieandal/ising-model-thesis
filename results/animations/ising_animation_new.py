import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

#### FUNCTIONS ####

lattice_n = 16
lattice_m = lattice_n
temp = 0.5
seed = 1
eq_steps = 5
J = 1
h = 0


def initial_state(N, M):   
    state = np.random.choice(np.array([-1,1]),size=(N,M))
    return state


def mcmove(lattice, inv_T):    
    '''Monte Carlo move using Metropolis algorithm '''
    n = lattice.shape[0]
    m = lattice.shape[1]
    for i in range(n):
        for j in range(m):
            # Periodicity for neighbors out of index
            ipp = (i + 1) if (i + 1) < n else 0
            jpp = (j + 1) if (j + 1) < m else 0
            inn = (i - 1) if (i - 1) >= 0 else (n - 1)
            jnn = (j - 1) if (j - 1) >= 0 else (m - 1)  
            
            # Calculate neighbors
            nb = lattice[ipp,j] + lattice[i,jpp] + lattice[inn,j] + lattice[i,jnn]
            
            # Compute energy difference
            spin =  lattice[i, j]
            deltaE = -2*spin*(J*nb + h)
            if deltaE >= 0:
                lattice[i, j] = -spin
            elif np.random.rand() < np.exp(deltaE*inv_T):
                lattice[i, j] = -spin
    return lattice

#### IMAGE APPEND ####

np.random.seed(seed) # set the random seed

img = [] # some array of images

lattice = initial_state(lattice_n, lattice_m)
img.append(lattice.copy())

inv_T = 1/(temp)

for i in range(eq_steps):
     for i in range(lattice_n):
        for j in range(lattice_m):
            # Periodicity for neighbors out of index
            ipp = (i + 1) if (i + 1) < lattice_n else 0
            jpp = (j + 1) if (j + 1) < lattice_m else 0
            inn = (i - 1) if (i - 1) >= 0 else (lattice_n - 1)
            jnn = (j - 1) if (j - 1) >= 0 else (lattice_m - 1)  
            
            # Calculate neighbors
            nb = lattice[ipp,j] + lattice[i,jpp] + lattice[inn,j] + lattice[i,jnn]
            
            # Compute energy difference
            spin =  lattice[i, j]
            deltaE = -2*spin*(J*nb + h)
            if deltaE >= 0:
                lattice[i, j] = -spin
            elif np.random.rand() < np.exp(deltaE*inv_T):
                lattice[i, j] = -spin
            img.append(lattice.copy())


#### PLOT ####

plt.imshow(img[0], vmin=-1, vmax=1, cmap='gray')
plt.show()

plt.imshow(img[-1], vmin=-1, vmax=1, cmap='gray')

plt.show()


#### ANIMATIONS ####

frames = [] # for storing the generated images
fig = plt.figure()
for i in range(eq_steps*lattice_n*lattice_m):
    frames.append([plt.imshow(img[i], cmap=cm.Greys_r, animated=True, vmin=-1, vmax=1)])

ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                repeat_delay=1000)
ani.save(f'metropolis_{lattice_n}_{temp}.mp4', dpi=300)
plt.show()


#### FUNCTIONS ####

lattice_n = 16
lattice_m = lattice_n
temp = 3.0
seed = 1
eq_steps = 20
J = 1
h = 0

def generate_lattice(N, M):
    return np.random.choice(np.array([-1,1], dtype=np.int8),size=(N,M))

def generate_array(N, M):
    return np.random.rand(N, M)

def mcmove(lattice, op_lattice, randvals, is_black, inv_temp):    
    '''Monte Carlo move using Metropolis algorithm '''
    n,m = lattice.shape
    for i in prange(n):
        for j in prange(m):
            # Set stencil indices with periodicity
            ipp = (i + 1) if (i + 1) < n else 0
            jpp = (j + 1) if (j + 1) < m else 0
            inn = (i - 1) if (i - 1) >= 0 else (n - 1)
            jnn = (j - 1) if (j - 1) >= 0 else (m - 1)  
           
            # Select off-column index based on color and row index parity
            if (is_black):
                joff = jpp if (i % 2) else jnn
            else:
                joff = jnn if (i % 2) else jpp
                
            # Compute sum of nearest neighbor spins
            nn_sum = op_lattice[inn, j] + op_lattice[i, j] + op_lattice[ipp, j] + op_lattice[i, joff]
            
            # Compute sum of nearest neighbor spins (taking values from neighboring
            spin_i = lattice[i, j]
            deltaE = -2* spin_i*(J*nn_sum+h)
            if deltaE >= 0:
                lattice[i, j] = -spin_i
            elif randvals[i, j] < math.exp(deltaE*inv_temp):
                lattice[i, j] = -spin_i
    return lattice

def combine_lattice(lattice_black, lattice_white):
    lattice = np.zeros((lattice_n, lattice_m), dtype=np.int8)
    for i in range(lattice_n):
        for j in range(lattice_m // 2):
            if (i % 2):
                lattice[i, 2*j+1] = lattice_black[i, j]
                lattice[i, 2*j] = lattice_white[i, j]
            else:
                lattice[i, 2*j] = lattice_black[i, j]
                lattice[i, 2*j+1] = lattice_white[i, j]
    return lattice



#### IMAGE APPEND ####

np.random.seed(seed) # set the random seed

img = [] # some array of images

lattice_black = np.ones((lattice_n, lattice_m//2))
lattice_white = np.ones((lattice_n, lattice_m//2))
# lattice_black = generate_lattice(lattice_n, lattice_m//2)
# lattice_white = generate_lattice(lattice_n, lattice_m//2)
lattice = combine_lattice(lattice_black, lattice_white)
img.append(lattice.copy())

img.append(combine_lattice(lattice_black, lattice_white).copy())
inv_temp = 1/(temp)

for i in range(eq_steps):
    # Black:
    randvals = generate_array(lattice_n, lattice_m//2)
    for i in range(lattice_n):
        for j in range(lattice_m//2):
            # Set stencil indices with periodicity
            ipp = (i + 1) if (i + 1) < lattice_n else 0
            jpp = (j + 1) if (j + 1) < lattice_m//2 else 0
            inn = (i - 1) if (i - 1) >= 0 else (lattice_n - 1)
            jnn = (j - 1) if (j - 1) >= 0 else (lattice_m//2 - 1)  
            
            # Select off-column index based on color and row index parity
            if (True):
                joff = jpp if (i % 2) else jnn
            else:
                joff = jnn if (i % 2) else jpp
                
            # Compute sum of nearest neighbor spins
            nn_sum = lattice_white[inn, j] + lattice_white[i, j] + lattice_white[ipp, j] + lattice_white[i, joff]
            
            # Compute sum of nearest neighbor spins (taking values from neighboring
            spin_i = lattice_black[i, j]
            deltaE = -2* spin_i*(J*nn_sum+h)
            if deltaE >= 0:
                lattice_black[i, j] = -spin_i
            elif randvals[i, j] < np.exp(deltaE*inv_temp):
                lattice_black[i, j] = -spin_i
    img.append(combine_lattice(lattice_black, lattice_white).copy())
    
    # White:
    randvals = generate_array(lattice_n, lattice_m//2)
    for i in range(lattice_n):
        for j in range(lattice_m//2):
            # Set stencil indices with periodicity
            ipp = (i + 1) if (i + 1) < lattice_n else 0
            jpp = (j + 1) if (j + 1) < lattice_m//2 else 0
            inn = (i - 1) if (i - 1) >= 0 else (lattice_n - 1)
            jnn = (j - 1) if (j - 1) >= 0 else (lattice_m//2 - 1)  
            
            # Select off-column index based on color and row index parity
            if (False):
                joff = jpp if (i % 2) else jnn
            else:
                joff = jnn if (i % 2) else jpp
                
            # Compute sum of nearest neighbor spins
            nn_sum = lattice_black[inn, j] + lattice_black[i, j] + lattice_black[ipp, j] + lattice_black[i, joff]
            
            # Compute sum of nearest neighbor spins (taking values from neighboring
            spin_i = lattice_white[i, j]
            deltaE = -2* spin_i*(J*nn_sum+h)
            if deltaE >= 0:
                lattice_white[i, j] = -spin_i
            elif randvals[i, j] < np.exp(deltaE*inv_temp):
                lattice_white[i, j] = -spin_i
    img.append(combine_lattice(lattice_black, lattice_white).copy())


plt.imshow(lattice_black, cmap='gray', vmin=-1, vmax=1)
plt.colorbar()
plt.show()
plt.imshow(lattice_white, cmap='gray', vmin=-1, vmax=1)
plt.colorbar()
plt.show()

plt.imshow(img[0], vmin=-1, vmax=1, cmap='gray')
plt.show()

plt.imshow(img[2], vmin=-1, vmax=1, cmap='gray')
plt.show()

#### ANIMATIONS ####

frames = [] # for storing the generated images
fig = plt.figure()
for i in range((eq_steps+1)*2):
    frames.append([plt.imshow(img[i], cmap=cm.Greys_r, animated=True, vmin=-1, vmax=1)])

ani = animation.ArtistAnimation(fig, frames, interval=500, blit=True,
                                repeat_delay=1000)
ani.save(f'checkerboard_{lattice_n}_{temp}.mp4', dpi=300)
plt.show()