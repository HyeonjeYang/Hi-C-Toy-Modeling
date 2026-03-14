#!/usr/bin/env python
# coding: utf-8

# Polychrom Customizing Simulation\
# last updated : 20260224

# In[1]:


import polychrom
#print(polychrom.__file__)
#check if you have polychrom, if not, please cp to your appropriate directory
#https://github.com/open2c/polychrom.git


# In[2]:


#from polychrom import contactmaps
#print(contactmaps.__file__)
#check the path


# In[3]:


import os
import sys
import logging


# In[4]:


import datetime


# In[5]:


from tqdm import tqdm


# In[6]:


#print(os.getcwd()) #check your current cwd


# In[7]:


import openmm #you mush have openmm to run the simulation

import polychrom
from polychrom import forcekits, forces, simulation, starting_conformations
from polychrom.hdf5_format import HDF5Reporter


# In[8]:


def create_polymer(N, polymer_creation_func, method, boxSize=100, r1=10, r2=13, step_size=1.0):

    """
    Polymer Generator
    """

    if polymer_creation_func == "grow_cubic":
        polymer = starting_conformations.grow_cubic(N, boxSize, method=method)
            #boxSize : cubic place to grow polymers. size of boxSize**3
            #methods : standard, linear, extended
            #standard : closed chain, ring polymer (ring)
            #linear : open chain, not connected (not a ring)
            #extended : long loop (ring)
            #we do not consider the strand crossing due to Topoisomerase II in this grow_cubic simulation
        return polymer

    elif polymer_creation_func == "create_spiral":
        polymer = starting_conformations.create_spiral(r1, r2, N)
        return polymer

    elif polymer_creation_func == "create_random_walk":
        polymer = starting_conformations.create_random_walk(step_size, N)
        return polymer

    elif polymer_creation_func == "create_constrained_random_walk":
        pass
        #work in progress

    else:
        raise ValueError(f"Unknown polymer_creation_func: {polymer_creation_func}")


# In[9]:


def add_confinement(sim, polymer_force_conf, density, k, r):
    if polymer_force_conf == "spherical_confinement":
        sim.add_force(forces.spherical_confinement(sim, density=density, k=k))
            #forces.spherical_confinement(sim, r="density", density=0.85, k=1)
            #~ k is the spring constant, the steepness of the wall (nucleus)
            #r is the radius, the default is density, you can change it manually (unit of nm)

    elif polymer_force_conf == "cylindrical_confinement":
        sim.add_force(forces.cylindrical_confinement(sim, k=k, bottom = 0, top=50, r=r))
    #cylindrical_confinement(sim_object, r, bottom=None, k=0.1, top=9999)

    else:
        raise ValueError(f"Unknown polymer_force_conf: {polymer_force_conf} or maybe working in progress, please try another force")


# In[10]:


def make_chains_eq(N, number_of_chains, is_ring=False):
    chains = []

    base = N // number_of_chains
    remainder = N % number_of_chains

    start = 0

    for i in range(number_of_chains):
        size = base
        if i == number_of_chains - 1:
            size += remainder

        end = start + size

        if i == number_of_chains - 1:
            chains.append((start, None, is_ring))
        else:
            chains.append((start, end, is_ring))

        start = end

    return chains


# In[11]:


def add_polymer_physics(sim, number_of_chains = 1, is_ring = False, trunc = 3.0, angle_k = 1.5, bond_length = 1.0, bond_wiggle = 0.05):

    N = sim.N
    chains = make_chains_eq(N, number_of_chains, is_ring)
    sim.add_force(
        forcekits.polymer_chains(
            sim,
            chains = chains,
            #chains=[(start, end, isRing)]
            #ende == None --> from start to the final one (0 to N-1)
            #isRing == False --> linear, not a ring / isRing == True --> ring
            # By default the library assumes you have one polymer chain
            # If you want to make it a ring, or more than one chain, use self.setChains
            # self.setChains([(0,50,True),(50,None,False)]) will set a 50-monomer ring and a chain from 50 to the end

            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={
                "bondLength": 1.0,
                "bondWiggleDistance": 0.05,  # Bond distance will fluctuate +- 0.05 on average
            },
            angle_force_func=forces.angle_force,
            angle_force_kwargs={
                "k": 1.5,
                # K is more or less arbitrary, k=4 corresponds to presistence length of 4,
                # k=1.5 is recommended to make polymer realistically flexible; k=8 is very stiff
            },
            nonbonded_force_func=forces.polynomial_repulsive, #only repulsive energy ~ nonboundedForce = forces.selective_SSW, keyworkds / make compartments!
            nonbonded_force_kwargs={
                "trunc": trunc,  # this will let chains cross sometimes
                # 'trunc':10.0, # this will resolve chain crossings and will not let chain cross anymore
            },
            except_bonds=True, #if False, then neighbour beads connected to the bond would gather and become so compact... (due to non-bonded forces)
            extra_bonds=None, #or you can put list [(i,j)] of extra bonds
            extra_triplets=None,
            override_checks=False
        )
    )

#making the physics of DNA


# In[12]:


def make_simulation(
    N,
    polymer_creation_func,
    method,
    r,
    number_of_chains = 1,
    boxSize=100,
    trunc=3.0,
    density=0.85,
    angle_k=1.5,
    is_ring=False,
    k=1.0,
    bond_length=1.0,
    bond_wiggle=0.05,
    polymer_force_conf="spherical_confinement",
    max_data_length = 5
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"trajectory_cylin_cubic_{timestamp}"
    reporter = HDF5Reporter(folder=folder_name, max_data_length = max_data_length, overwrite=True)

    """
    Basic Parameters for the Simulation
    """
    sim = simulation.Simulation(
        platform="CUDA",
        integrator="variablelangevin",  #variablelangevin, langevinmiddle, verlet, variableverlet, brownian
        error_tol=0.03,
        GPU="1",
        collision_rate=0.04, #0.03 recommended
        N=N,
        save_decimals=2,
        PBCbox=False, #False --> infinite space, like packman effect.
        reporters=[reporter],
        precision="single", #single, mixed, double / but single recommended. double is really slow (10 times slower...)
        max_Ek=10, #raise error if avg kinetic energy exceeds this value ~ <E> > max_Ek --> raises errorr
        temperature=300, #in the unit of Kelvin #default = 300
        verbose=False, #recommend not to change... if True: --> some many stuffs will come out.
        length_scale=1.0, #scaling factor of the system. length_scale of 1.0 --> harmonic bond has the length scale of 1 nm
        mass=100, #unit : amu, sometimes it can be effective according to Raynolds number.
    )

    """
    Polymer Creation
    """

    polymer = create_polymer(N, polymer_creation_func, method, boxSize=100, r1=10, r2=13, step_size=1.0)

    """
    Data Setting ~ load(s) (a) polymer(s)
    """

    sim.set_data(data=polymer, center=True, random_offset=1e-5, report=True)  #it loads a polymer (set particle positions)
    #center = False / True / "zero"
    #True : move center of mass to zero
    #False : ~ True
    #"zero" : then center the data such as all positions are positive and start at zero

    #random_offset : a little noise

    #sim.set_velocities(v)
    #Langevin and Brownian --> it automatically sets the initial velocity. So don't worry.

    """
    Confinement
    """

    add_confinement(sim, polymer_force_conf, density=density, k=k, r=r)

    """
    Polymer Physics
    """
    add_polymer_physics(sim, is_ring = is_ring, trunc = trunc, angle_k = angle_k, bond_length = bond_length, bond_wiggle = bond_wiggle)

    assert polymer.shape[0] == N, "Polymer length does not match N"

    return sim, reporter #return the polymer


# polymer : N x 3 array of the polymer. (coordinates included) ~ array of positions

# In[16]:


#choose the number of monomers
N = 50000
is_ring = False #if False : not a ring, if True : a linear chromosome
trunc = 3.0 #if low, then topoisomerase-like, if high, then no strand crossing

#choose a function for your polymer creation
polymer_creation_func = "grow_cubic" #you can change it manually
#create_spiral(r1, r2, N) : easy mitotic like starting conformation ~ like a sphiral confinement
#create_random_walk(step_size, N) : constrained freely joined chain of length N
#create_constrained_random_walk(N, constraint_f, starting_point, step_size, polar_fixed) : similar to create_random_walk, but with constraint functions
#grow_cubic(N, boxSize, method) : ring / linear polymer on a cubic lattice
method = "standard"
boxSize = 500

#confinement
polymer_force_conf = "cylindrical_confinement" #making the space!
#spherical_confinement
#cylindrical_confinement
#usually used for nucleus modeling

#spherical_well
#tether_particles
#pull_force
density = 0.2 #0.85 recommended
k = 1.0 #1.0 recommended
r = 5.0

#polymer physics
angle_k = 1.5
bond_length = 1.0 #nm
bond_wiggle = 0.05

#equal-partition
number_of_chains = 1

#max_data_length ~ save every {max_data_length} blocks
max_data_length = 1000

#binsize = 4


# In[17]:


sim, reporter = make_simulation(
    N = N,
    polymer_creation_func = polymer_creation_func,
    method = method,
    boxSize = boxSize,
    trunc = trunc,
    density = density,
    angle_k = angle_k,
    is_ring = is_ring,
    k = k,
    r = r,
    bond_length = bond_length,
    bond_wiggle = bond_wiggle,
    polymer_force_conf = polymer_force_conf,
    max_data_length = max_data_length
)


# In[15]:


#number of steps
block_steps = 5000
#timesteps ~ block_timesteps per block
block_timesteps = 50 # ~ 100 integration steps


# In[17]:


sim.local_energy_minimization(tolerance = 1e-2, maxIterations = 100)


# In[18]:


logging.getLogger().setLevel(logging.WARNING)

for _ in tqdm(range(block_steps), file=sys.stdout, mininterval=1): # Do block_steps blocks
    #I recommend that you not change "leave = False" to "leave = "True"
    sim.do_block(block_timesteps) # Of block_timesteps timesteps each. Data is saved automatically.


sim.print_stats()  # In the end, print very simple statistics
reporter.dump_data()  # always need to run in the end to dump the block cache to the disk


# In[ ]:


#let's make Hi-C and do the visualization with Blender!!!
#Yeah~


# In[ ]:




