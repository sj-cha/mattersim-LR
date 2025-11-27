from ase.io import read, write
from ase import units
from ase.md.langevin import Langevin
from ase.md.bussi import Bussi
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

import pandas as pd
from mattersim.forcefield import MatterSimCalculator

calculator = MatterSimCalculator.from_checkpoint(load_path='../../baseline/results/best_model.pth', device='cuda', long_range = False)

def print_step(dyn, temp, energy):
    atoms = dyn.atoms
    T_inst = atoms.get_temperature()
    E_kin = atoms.get_kinetic_energy()  # kinetic energy in eV
    E_pot = atoms.get_potential_energy()  # potential energy in eV
    E_tot = E_kin + E_pot

    print(f"Step: {dyn.nsteps}, Temperature: {T_inst:.2f} K, Total Energy: {E_tot:.6f} eV")
    
    temp.append(T_inst)
    energy.append(E_tot)
    
def write_xyz(filename, atoms):
    write(filename+ ".xyz", atoms, format="extxyz", append=True)

def MD(thermostat, atoms, T, timestep, time, interval, filename):
    if time < 1000:
        filename = f'{filename}_{T}_{time}fs'
    elif time < 1000000:
        filename = f'{filename}_{T}_{time/1000:.3f}ps'
    else:
        filename = f'{filename}_{T}_{time/1000000:.2f}ns'
    
    atoms.calc = calculator
    atoms.center()   

    temp = []
    energy = []
    
    MaxwellBoltzmannDistribution(atoms, temperature_K=T)
    Stationary(atoms)  
    ZeroRotation(atoms)

    if thermostat == "langevin":
        dyn = Langevin(
                atoms,
                timestep=timestep * units.fs,
                temperature_K=T,
                friction=0.01 / units.fs,
            )

    elif thermostat == "bussi":
        dyn = Bussi(
                atoms,
                timestep=timestep * units.fs,
                temperature_K=T,
                taut=timestep*40*units.fs,
            ) 
    
    dyn.attach(write_xyz, interval=interval, atoms=atoms, filename=filename)
    dyn.attach(lambda: print_step(dyn, temp, energy), interval=interval)

    open(filename + ".xyz", "w").close()
    dyn.run(round(time/timestep))
    
    df = pd.DataFrame({"T": temp, "E": energy})
    df.to_csv(f"{filename}.csv")

filename = "water"

# Load structure
atoms = read(f'../test_opt.xyz')
atoms.pbc = True
atoms.cell = [[13, 0, 0],[0, 13, 0],[0, 0, 13]]

MD(thermostat="langevin",
    atoms=atoms, 
    T=300, # K
    timestep=1, # fs
    time=100000, # fs
    interval=20, # save every n fs
    filename=f"{filename}")
