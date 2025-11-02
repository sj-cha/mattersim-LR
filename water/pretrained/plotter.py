# -*- coding: utf-8 -*-
import argparse
import os
import pickle as pkl
import random

import numpy as np
import torch
# import torch.distributed
import wandb
from ase.units import GPa

from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.m3gnet.scaling import AtomScaling
from mattersim.forcefield.potential import Potential
from mattersim.utils.atoms_utils import AtomsAdaptor
from mattersim.utils.logger_utils import get_logger

from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt

RESULTS_FOLDER = "figures"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def save_plot(filename):
    """Save current matplotlib figure into results folder."""
    path = os.path.join(RESULTS_FOLDER, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {path}")

atoms_val = AtomsAdaptor.from_file(filename="../../data/benchmarks/val_water_1593_eVAng.xyz")
energies = []
forces = [] 
for atoms in atoms_val:
    energies.append(atoms.get_potential_energy() / len(atoms)) 
    forces.append(atoms.get_forces())

print(f"Processed {len(atoms_val)} validation data")

val_dataloader = build_dataloader(
    atoms_val,
    energies,
    forces,
    pin_memory=True,
    is_distributed=False,
    shuffle=False,
    batch_size=16
)

potential = Potential.from_checkpoint(
    load_path="../../pretrained_models/mattersim-v1.0.0-1M.pth",
    load_training_state=False
)

predicted_energies, predicted_forces, _ = potential.predict_properties(
    val_dataloader,
    include_forces=True,
    include_stresses=False
)

predicted_energies = np.array(predicted_energies) / len(atoms)

true_energies, true_forces = energies, forces

# ---- Parity Plots ----
energies_mae = mean_absolute_error(true_energies, predicted_energies)
energies_r2 = r2_score(true_energies, predicted_energies)
print(f"Energies MAE: {energies_mae:.4f}, R²: {energies_r2:.4f}")

forces_mae = mean_absolute_error(np.vstack(true_forces), np.vstack(predicted_forces))
forces_r2 = r2_score(np.vstack(true_forces), np.vstack(predicted_forces))
print(f"Forces MAE: {forces_mae:.4f}, R²: {forces_r2:.4f}")

# 1. Energy Parity Plot
plt.figure()
plt.scatter(true_energies, predicted_energies, color="purple", alpha=0.7, label="Data")
min_e, max_e = min(true_energies), max(true_energies)
plt.plot([min_e, max_e], [min_e, max_e], "k--", label="Ideal")
plt.xlabel("True Energies")
plt.ylabel("Predicted Energies")
plt.title(f"Energy Parity Plot\nMAE={energies_mae:.4f}, R²={energies_r2:.4f}")
plt.legend()
save_plot("parity_energy.png")
plt.close()

# 2. Forces Parity Plot
true_forces_flat = np.vstack(true_forces).flatten()
predicted_forces_flat = np.vstack(predicted_forces).flatten()

plt.figure()
plt.hexbin(true_forces, predicted_forces, gridsize=300, cmap='viridis', bins='log')
min_f, max_f = min(true_forces_flat), max(true_forces_flat)
plt.plot([min_f, max_f], [min_f, max_f], "k--", label="Ideal")
plt.xlabel("True Forces")
plt.ylabel("Predicted Forces")
plt.title(f"Forces Parity Plot\nMAE={forces_mae:.4f}, R²={forces_r2:.4f}")
plt.legend()
save_plot("parity_forces.png")
plt.close()