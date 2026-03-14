#!/usr/bin/env python
# coding: utf-8

# Modified from polychrom extrusion_3D\
# Original Source : https://github.com/open2c/polychrom.git \
# Licensed under MIT License\
# Copyright (c) 2019 Massachusetts Institute of Technology

# In[1]:


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


# In[2]:


from bondUpdater import bondUpdater


# In[3]:


input_folder_name = "trajectory_20260313_1"
h5_filepath = f"{input_folder_name}/LEFPositions_big.h5"
output_folder_name = "LEFPositions_simu_20260313_big_chromosome_1_restriction"


# In[ ]:


def generate_gaussian_spaced_list(end, start=0, mean_gap=2500, std_gap=1000):
    values = [start]

    while values[-1] < end:
        gap = np.random.normal(loc=mean_gap, scale=std_gap)

        # 음수 간격 방지
        if gap <= 0:
            continue

        next_value = values[-1] + gap

        if next_value > end:
            break

        values.append(int(round(next_value)))

    return values

# 실행
#result = generate_gaussian_spaced_list(end = N-1)


# In[4]:


def add_polymer_physics(sim, stickyParticlesIdxs, is_ring=False, trunc=3.0, angle_k=1.5, bond_length=1.0, bond_wiggle=0.1):
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
                'stickyParticlesIdxs' : stickyParticlesIdxs,
                'extraHardParticlesIdxs' : [], #여기에서 stickyParticlesIdxs랑 같이 맞춰도 됨.
                'repulsionRadius' : 1.5, #recommend greater than zero
                'attractionRadius' : 1.0, #how far the particle actually fill in the interaction region, it should be greater than 0.0
                'attractionEnergy' : 0.1, #attraction with all
                'selectiveAttractionEnergy' : 0.5 #lower than 1.0
                #'trunc': trunc, 
                #'radiusMult': 1.05,
            },
            except_bonds=True,
        )
    )


# In[6]:


def make_chains_eq(N, number_of_chains, is_ring):
    if number_of_chains == 1:
        return [(0, None, is_ring)]
    #for multiple chains
    return [(i * (N // number_of_chains), (i + 1) * (N // number_of_chains), is_ring) 
            for i in range(number_of_chains)]


# Run the Simulation!

# Basic Parameters

# In[7]:


logging.getLogger("polychrom").setLevel(logging.WARNING)


def run_experiment(folder_name, h5_filepath, params):

    cut_state = np.zeros(len(params["initial_data"]) - 1, dtype=bool)

    k_cut = params.get("k_cut", 1e-6)   # cut rate per bond
    dt_cut = params["steps"] * params["timestep"]       # timestep × MD steps
    p_cut = 1 - np.exp(-k_cut * dt_cut)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    # ---- cut fraction log file ----
    cut_log_path = os.path.join(folder_name, "cut_fraction_blocks.txt")
    cut_log_file = open(cut_log_path, "w")
    cut_log_file.write("fraction\tblock\n")

    next_threshold = 0.1

    # ---- load LEF positions ----
    with h5py.File(h5_filepath, mode="r") as f:
        lef_positions = f["positions"][:]

    milker = bondUpdater(lef_positions)

    reporter = HDF5Reporter(
        folder=folder_name,
        max_data_length=100,
        overwrite=True,
        blocks_only=False,
    )

    data = params["initial_data"]
    sim_inits_total = lef_positions.shape[0] // params["restart_every"]

    pbar = tqdm(total=sim_inits_total, desc="Total Progress")

    # =============================
    # Main simulation loop
    # =============================
    for iteration in range(sim_inits_total):

        print(f"\nIteration {iteration+1}/{sim_inits_total}")

        # ---- Simulation object ----
        sim = Simulation(
            platform="cuda",
            integrator="langevin",
            error_tol=0.01,
            GPU=params.get("GPU", "1"),
            collision_rate=params.get("collision_rate", 0.03),
            N=len(data),                         
            reporters=[reporter],
            PBCbox=params["box"],
            precision="mixed",
            timestep=params["timestep"], #이거 variableLangevin이면 삭제하기!
        )

        # ---- Load polymer ----
        sim.set_data(data)

        # ---- Add polymer physics ----
        add_polymer_physics(sim, params["stickyParticlesIdxs"])
        bond_force = sim.force_dict["harmonic_bonds"]

        # ---- SMC bond parameters ----
        kbond = sim.kbondScalingFactor / (params["smc_wiggle"] ** 2)

        bondDist = params["smc_dist"] * sim.length_scale  # 

        activeParams = {"length": bondDist, "k": kbond}
        inactiveParams = {"length": bondDist, "k": 0}

        milker.setParams(activeParams, inactiveParams)

        milker.setup(
            bondForce=sim.force_dict["harmonic_bonds"],
            blocks=params["restart_every"],
        )

        # ---- Initialization ----
        if iteration == 0:
            sim.local_energy_minimization()
        else:
            sim._apply_forces()

        # =============================
        # Block loop
        # =============================
        for i in range(params["restart_every"]):

            # --- Poisson restriction enzyme ---
            events = np.random.rand(N-1) < p_cut
            newcuts = np.where(events & (~cut_state))[0]

            if len(newcuts) > 0:
                for site in newcuts:

                    bond_force.setBondParameters(
                        site,
                        site,
                        site+1,
                        1.0,     # bond length
                        0.02     # 거의 0 stiffness #not complete bond removal
                    )

                bond_force.updateParametersInContext(sim.context)
                cut_state[newcuts] = True
                #relaxation step for energy stability
                sim.integrator.step(100)

            # --- check cut fraction ---
            fraction = np.sum(cut_state) / len(cut_state)

            while fraction >= next_threshold and next_threshold <= 1.0:
                cut_log_file.write(
                    f"{next_threshold:.2f}\t{iteration * params['restart_every'] + i}\n"
                  )
                cut_log_file.flush()
                next_threshold += 0.1

            if i % params["save_every"] == (params["save_every"] - 1):
                sim.do_block(steps=params["steps"])
            else:
                sim.integrator.step(params["steps"])

            if i < params["restart_every"] - 1:
                milker.step(sim.context)

        # ---- Save updated polymer ----
        data = sim.get_data()
        del sim

        reporter.blocks_only = True
        pbar.update(1)

        time.sleep(0.2)

    pbar.close()
    reporter.dump_data()
    
    cut_log_file.close()


# In[8]:


# -------defining parameters----------
#  -- basic loop extrusion parameters

myfile = h5py.File(h5_filepath, mode='r') #you have to create your "trajectory" folder and "LEFPosition.h5" in order to run the simulation

N = myfile.attrs["N"]
LEFNum = myfile.attrs["LEFNum"]
LEFpositions = myfile["positions"]

Nframes = LEFpositions.shape[0] #check the frames and get how many frames does LEFpositions have


steps = 500   # MD steps per step of cohesin
stiff = 1
dens = 0.1
box = ((N / dens) ** 0.33)  # density = 0.1.
data = grow_cubic(N, int(box) - 2)  # creates a compact conformation ~ for reference, see the polychrom.starting_conformations
block = 0  # starting block 

# new parameters because some things changed 
saveEveryBlocks = 10   # save every 10 blocks (saving every block is now too much almost)
restartSimulationEveryBlocks = 100

# parameters for smc bonds
smcBondWiggleDist = 0.2
smcBondDist = 0.5

# assertions for easy managing code below 
assert (Nframes % restartSimulationEveryBlocks) == 0 
assert (restartSimulationEveryBlocks % saveEveryBlocks) == 0

savesPerSim = restartSimulationEveryBlocks // saveEveryBlocks
simInitsTotal  = (Nframes) // restartSimulationEveryBlocks

stickyParticlesIdxs = generate_gaussian_spaced_list(N) #compartment 만드는 코드


# In[ ]:


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
        'angle_k': 1.5,
        'collision_rate': 0.3, #higher collision rate --> more stable (when 0.03, energy explodes!)
        'temperature': 300,
        'integrator': 'langevin',

        #stickyParticleIdxs
        'stickyParticlesIdxs' : stickyParticlesIdxs,

        #k_cut ~ 1 - exp(-k_cut delta t) ~ rate const of Poisson Process
        'k_cut': 5e-7,

        'timestep' : 10
    }

    run_experiment(output_folder_name, h5_filepath, my_params)




# In[ ]:


#check for iterations
with h5py.File(h5_filepath, mode='r') as f:
    total_lef_steps = f["positions"].shape[0]

print(f"Total LEF Steps: {total_lef_steps}")
print(f"Restart Every: {my_params['restart_every']}")
print(f"Total Iteration: {total_lef_steps // my_params['restart_every']}")


# In[ ]:


print(Nframes)


# In[ ]:


#40, 50 for fixed Langevin

#처음 polymer는 있고 restriction, TADs, compartments의 경쟁 모델, not equilibrium

#cut 직후에 잠깐 relaxation