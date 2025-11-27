from mattersim.forcefield import MatterSimCalculator
from ase.io import read, write
from ase.optimize import LBFGS

calculator = MatterSimCalculator.from_checkpoint(load_path='../hard_const/results/best_model.pth', device='cuda', long_range = True)

atoms = read("./solvation_zn.xyz")
atoms.pbc = True
atoms.cell = [[13, 0, 0],[0, 13, 0],[0, 0, 13]]
atoms.calc = calculator
opt = LBFGS(atoms)
opt.run(fmax=0.05)

write("solvation_zn_opt.xyz", atoms)