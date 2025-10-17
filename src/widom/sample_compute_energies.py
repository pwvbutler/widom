# Copyright (c) 2025 CuspAI
# All rights reserved.

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from tqdm_loggable.auto import tqdm

from widom.utils import check_accessibility, create_supercell_if_needed, sample_gas_positions, generate_grid_positions

def sample_compute_energies(
    calculator: Calculator,
    structure: Atoms,
    gas: Atoms,
    num_insertions: int,
    cutoff_distance: float,
    cutoff_to_com: bool,
    min_interplanar_distance: float,
    max_distance_to_host: float,
    use_grid: bool,
    grid_spacing: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the Widom insertion algorithm to calculate interaction energies.

    This function performs Widom insertion of gas molecules into a structure to calculate
    interaction energies, which can be used for computing the Henry coefficient and heat
    of adsorption.

    Args:
        calculator: Calculator object from ASE to calculate energies.
        structure: Atoms object representing the framework structure.
        gas: Atoms object representing the gas molecule to insert.
        num_insertions: Number of random insertions to perform.
        cutoff_distance: Minimum allowed distance between framework atoms and gas molecule, in angstroms.
        cutoff_to_com: Whether to use the center of mass for distance calculations.
        min_interplanar_distance: Minimum interplanar distance before constructing a supercell, in angstroms.
        max_distance_to_host: Maximum distance from gas to host atoms.
        use_grid: Whether to use a grid-based sampling of positions.
        grid_spacing: Spacing of the grid in angstroms.
        random_seed: Seed for random number generator to ensure reproducibility.

    Returns:
        A tuple containing:
            - Array of total energies (in eV) for each insertion attempt.
            - Array of booleans indicating whether the insertion was successful (non-overlapping with framework atoms).
            - Array of positions of the inserted gas molecules.
    """
    # Make copies to avoid modifying input
    structure = structure.copy()
    gas = gas.copy()

    # Supercell if needed
    structure = create_supercell_if_needed(structure, min_interplanar_distance)

    if use_grid:
        pos_grid, accessible_pos = generate_grid_positions(
            structure,
            grid_spacing=grid_spacing,
            cutoff_distance=cutoff_distance,
            max_distance=max_distance_to_host,
            min_interplanar_distance=min_interplanar_distance,
        )
        print(f"Accessible grid positions: {len(accessible_pos)} / {len(pos_grid)}")

        # sample grid positions with replacement
        rng = np.random.default_rng(random_seed)
        chosen_idx = rng.choice(len(accessible_pos), size=num_insertions, replace=True)
        chosen_positions = accessible_pos[chosen_idx]

        # Place gas COM at chosen positions
        gas_positions = np.zeros((num_insertions, len(gas), 3))
        for i, pos in enumerate(chosen_positions):
            added_gas = gas.copy()
            added_gas.cell = structure.cell
            added_gas.pbc = structure.pbc

            # randomly rotate gas molecule
            angle = rng.random() * 360
            axis = rng.random(3)
            axis /= np.linalg.norm(axis)
            added_gas.rotate(v=axis, a=angle)

            # translate to grid position
            added_gas.translate(pos)

            # Wrap into cell
            added_gas.wrap()
            gas_positions[i] = added_gas.get_positions()

        is_accessible = np.ones(num_insertions, dtype=bool)

    else:
        # Fallback to random sampling
        rng = np.random.default_rng(random_seed)
        gas_positions = sample_gas_positions(structure, gas, num_insertions, rng)
        framework_coords = structure.get_positions()
        lattice_matrix = np.array(structure.cell)
        is_accessible = check_accessibility(
            gas_positions,
            framework_coords,
            lattice_matrix,
            cutoff_distance,
            cutoff_to_com,
            max_distance=max_distance_to_host,
        )

        print(f"Number of accessible positions: {np.sum(is_accessible)} out of {num_insertions}")

    # Prepare arrays for results
    energies = np.zeros(num_insertions)  # [eV], total energy

    # Set inaccessible positions to high energy
    energies[~is_accessible] = 1e10

    # Batch evaluate energies for accessible positions only
    accessible_indices = np.where(is_accessible)[0]
    structure_with_gas_original = structure + gas
    for i in tqdm(accessible_indices):
        structure_with_gas = structure_with_gas_original.copy()
        structure_with_gas.arrays["positions"][-len(gas) :] = gas_positions[i]
        structure_with_gas.wrap()  # wrap atoms to unit cell
        structure_with_gas.calc = calculator
        energies[i] = structure_with_gas.get_potential_energy()  # [eV]

    return energies, is_accessible, gas_positions  # [eV]