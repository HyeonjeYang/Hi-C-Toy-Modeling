#!/usr/bin/env python
# coding: utf-8

# Modified from polychrom extrusion_3D\
# Original Source : https://github.com/open2c/polychrom.git \
# Licensed under MIT License\
# Copyright (c) 2019 Massachusetts Institute of Technology

# In[70]:


import pickle
import os
import time
import numpy as np
import polychrom

from polychrom import polymerutils
from polychrom import forces
from polychrom import forcekits
from polychrom.simulation import Simulation
from polychrom.starting_conformations import grow_cubic
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file

import openmm 
import os 
import shutil


import warnings
import h5py 
import glob

import logging
import sys
from tqdm import tqdm
from IPython.display import clear_output


# In[71]:


from bondUpdater import bondUpdater


# In[72]:


input_folder_name = "trajectory_20260225_2"
h5_filepath = f"{input_folder_name}/LEFPositions_big.h5"
output_folder_name = "LEFPositions_simu_20260225_big_chromosome_2"


# In[73]:


def add_polymer_physics(sim, is_ring=False, trunc=3.0, angle_k=1.5, bond_length=1.0, bond_wiggle=0.1):
    """
    polymer_physics
    """
    sim.add_force(
        forcekits.polymer_chains(
            sim,
            chains=[(0, None, is_ring)],
            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={
                'bondLength': bond_length,
                'bondWiggleDistance': bond_wiggle,
            },
            angle_force_func=forces.angle_force,
            angle_force_kwargs={
                'k': angle_k
            },
            nonbonded_force_func=forces.selective_SSW,
            nonbonded_force_kwargs={
                'stickyParticlesIdxs' : list(range(500, 1000)) +list(range(3000, 3500)) + list(range(4200, 4900)),
                'extraHardParticlesIdxs' : [],
                'repulsionRadius' : 0.0,
                'attractionRadius' : 0.0,
                #'trunc': trunc, 
                #'radiusMult': 1.05,
            },
            except_bonds=True,
        )
    )


# In[74]:


def make_chains_eq(N, number_of_chains, is_ring):
    if number_of_chains == 1:
        return [(0, None, is_ring)]
    #for multiple chains
    return [(i * (N // number_of_chains), (i + 1) * (N // number_of_chains), is_ring) 
            for i in range(number_of_chains)]


# In[75]:


logging.getLogger("polychrom").setLevel(logging.WARNING)

def run_experiment(folder_name, h5_filepath, params):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    with h5py.File(h5_filepath, mode='r') as f:
        lef_positions = f["positions"][:]
        N = f.attrs["N"]

    milker = bondUpdater(lef_positions)
    reporter = HDF5Reporter(folder=folder_name, max_data_length=100, overwrite=True)

    data = params['initial_data']
    sim_inits_total = lef_positions.shape[0] // params['restart_every']

    pbar = tqdm(total=sim_inits_total, desc="Total Progress")

    #main simulation loop
    for iteration in range(sim_inits_total):

        clear_output(wait=True)

        pbar.display()
        print(f"Current Progress: {iteration + 1} / {sim_inits_total} calculation in progress...")

        sim = Simulation(
            platform="cuda", GPU="0", integrator=params.get('integrator','variableLangevin'),
            error_tol=0.01, N=N, reporters=[reporter],
            PBCbox=params['box'], precision="mixed", collision_rate=params.get('collision_rate', 0.02), 
            temperature=params.get('temperature', 300),
            verbose=False
        ) #you can actually change the parameters

        sim.set_data(data)

        add_polymer_physics(
            sim, 
            angle_k=params.get('angle_k', 1.5),
            trunc=params.get('trunc', 1.5)
        )

        kbond = sim.kbondScalingFactor / (params['smc_wiggle'] ** 2)
        milker.setParams({"length": params['smc_dist'], "k": kbond}, 
                         {"length": params['smc_dist'], "k": 0})
        milker.setup(bondForce=sim.force_dict['harmonic_bonds'], blocks=params['restart_every'])

        if iteration == 0:
            sim.local_energy_minimization()
        #sim.context.setVelocitiesToTemperature(params.get('temperature', 300))
        else: sim._apply_forces()

        #calculation loop
        for i in range(params['restart_every']):
            if i % params['save_every'] == (params['save_every'] - 1):
                sim.do_block(steps=params['steps'])
            else:
                sim.integrator.step(params['steps'])

            if i < params['restart_every'] - 1:
                milker.step(sim.context)
                sim.integrator.step(10)

        data = sim.get_data()
        del sim
        reporter.blocks_only = True

        pbar.update(1)
        time.sleep(0.2)

    pbar.close()
    reporter.dump_data()


# Run the Simulation!

# Basic Parameters

# In[76]:


# -------defining parameters----------
#  -- basic loop extrusion parameters

myfile = h5py.File(h5_filepath, mode='r') #you have to create your "trajectory" folder and "LEFPosition.h5" in order to run the simulation

N = myfile.attrs["N"]
LEFNum = myfile.attrs["LEFNum"]
LEFpositions = myfile["positions"]

Nframes = LEFpositions.shape[0] #check the frames and get how many frames does LEFpositions have


steps = 200   # MD steps per step of cohesin
stiff = 1
dens = 0.3
box = 10*( (N / dens) ** 0.33)  # density = 0.1.
data = grow_cubic(N, int(box) - 2)  # creates a compact conformation ~ for reference, see the polychrom.starting_conformations
block = 0  # starting block 

# new parameters because some things changed 
saveEveryBlocks = 50   # save every 10 blocks (saving every block is now too much almost)
restartSimulationEveryBlocks = 100

# parameters for smc bonds
smcBondWiggleDist = 1.0
smcBondDist = 1.5

# assertions for easy managing code below 
assert (Nframes % restartSimulationEveryBlocks) == 0 
assert (restartSimulationEveryBlocks % saveEveryBlocks) == 0

savesPerSim = restartSimulationEveryBlocks // saveEveryBlocks
simInitsTotal  = (Nframes) // restartSimulationEveryBlocks


# In[56]:


if __name__ == "__main__":

    my_params = {
        'initial_data': data,  # grow_cubic로 생성한 데이터

        # box size: 위에서 계산한 값 사용
        'box': [box, box, box],

        # 위에서 정의한 simulation control 파라미터 사용
        'restart_every': restartSimulationEveryBlocks,
        'save_every': saveEveryBlocks,
        'steps': steps,

        # SMC bond parameters (위에서 정의한 값)
        'smc_wiggle': smcBondWiggleDist,
        'smc_dist': smcBondDist,

        # polymer physics
        'angle_k': 1.0,
        'collision_rate': 2.0,
        'temperature': 290,
        'integrator': 'variableLangevin',
    }

    run_experiment(output_folder_name, h5_filepath, my_params)


# In[58]:


#check for iterations
with h5py.File(h5_filepath, mode='r') as f:
    total_lef_steps = f["positions"].shape[0]

print(f"Total LEF Steps: {total_lef_steps}")
print(f"Restart Every: {my_params['restart_every']}")
print(f"Total Iteration: {total_lef_steps // my_params['restart_every']}")


# In[59]:


print(Nframes)


# In[ ]:




