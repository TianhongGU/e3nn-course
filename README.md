# Equivariant Graph Neural Networks with e3nn — A Hands-On Course

A comprehensive, student-friendly tutorial series that builds up from the mathematical
foundations of Euclidean symmetry to complete, working implementations of state-of-the-art
equivariant interatomic potentials: **NequIP**, **Allegro**, and **MACE** — with invariant
baselines (**SchNet**, **DimeNet**) along the way.

Every lesson follows the same philosophy:

1. **Theory first.** Each operation is derived and motivated mathematically
   *before* presenting any code. We attempt to cite the key manuscripts
   alongside equation numbers therein.
2. **Theory and experiment, side-by-side.** Implementations are broken into code
   blocks of small to moderate sizes while being connected to their
   corresponding equations.
3. **Verifications.** Every key equivariant operation is unit-tested for
   numerically, using shared helper functions from the local `course_utils`
   folder. Feel free to inspect and modify them at your own convenience!
4. **Visualizations.** Spherical harmonics, tensor-product selection rules,
   learned features, training curves, and MD trajectories are plotted throughout
   to assist the reader in building intuition and gaining visual insight.
5. **Modularity and simplicity.** Long lessons are split into `*_a`, `*_b`,
   `*_c` notebooks that can be followed in one sitting, individually.

## Prerequisites

- Linear algebra (matrices, eigenvalues, change of basis), basic group theory
  helps but is introduced from scratch in Lesson 01.
- PyTorch basics (tensors, autograd, `nn.Module`, training loops).
- Some exposure to molecular systems / atomistic simulation is helpful for Parts
  03-05 but not required.

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
cd e3nn_course
uv sync                          # creates .venv with all dependencies
.venv/bin/python -m ipykernel install --user --name e3nn-course
```

Key packages: `torch`, `e3nn`, `torch-geometric`, `ase`, `matplotlib`, `plotly`.

## Curriculum Map

### Part I — Foundations: Symmetry, Irreps, and Equivariant Operations (Lessons 01–04)

| Notebook | Topic |
|---|---|
| `01a_symmetry_and_equivariance.ipynb` | Why symmetry? Groups, E(3)/SE(3)/O(3), invariance vs. equivariance, why data augmentation is not enough. |
| `01b_group_representations.ipynb` | Representations, reducibility, Schur's lemma in practice, Wigner D-matrices. |
| `02a_irreps_in_e3nn.ipynb` | `e3nn.o3.Irreps`: the type system of equivariant networks — scalars `0e`, vectors `1o`, parity, direct sums. |
| `02b_spherical_harmonics.ipynb` | Spherical harmonics as the equivariant embedding of directions; visualization; `o3.spherical_harmonics`. |
| `03a_tensor_products_theory.ipynb` | Coupling irreps: Clebsch–Gordan coefficients, selection rules, why the tensor product is *the* equivariant bilinear operation. |
| `03b_tensor_products_e3nn.ipynb` | `o3.FullyConnectedTensorProduct` & friends: paths, weights, instructions dissected. |
| `04_nonlinearities_and_gates.ipynb` | Equivariant nonlinearities: norm activations, `e3nn.nn.Gate`; building an equivariant MLP. |

### Part II — From Operations to Networks (Lessons 05–06)

| Notebook | Topic |
|---|---|
| `05a_atomistic_graphs.ipynb` | Point clouds → graphs: cutoffs, neighbor lists, periodic boundary conditions (ASE + torch-geometric). |
| `05b_radial_basis_and_cutoffs.ipynb` | Radial basis functions (Bessel, Gaussian), envelope/cutoff functions, smoothness requirements for potentials. |
| `06a_equivariant_convolution.ipynb` | The equivariant graph convolution (Tensor Field Networks / e3nn point convolution): equations + implementation. |
| `06b_tetris_end_to_end.ipynb` | The classic e3nn "Tetris" exercise: classify chiral 3D shapes with a small equivariant GNN; demonstrate parity. |

### Part III — Invariant Baselines (Lesson 07)

| Notebook | Topic |
|---|---|
| `07a_schnet.ipynb` | SchNet: continuous-filter convolutions; the invariant message-passing blueprint. |
| `07b_dimenet.ipynb` | DimeNet(++): directional message passing with angles; strengths and the incompleteness problem. |

### Part IV — State-of-the-Art Equivariant Potentials (Lessons 08–10)

| Notebook | Topic |
|---|---|
| `08a_nequip_theory.ipynb` | NequIP: architecture equations, interaction blocks, why l>0 features boost data efficiency. |
| `08b_nequip_implementation.ipynb` | Block-by-block NequIP in e3nn; equivariance tests; training on a small dataset. |
| `09a_allegro_theory.ipynb` | Allegro: strictly local equivariant descriptors, the scalar track / tensor track design, scalability arguments. |
| `09b_allegro_implementation.ipynb` | Block-by-block Allegro; comparison with NequIP on the same data. |
| `10a_ace_theory.ipynb` | Atomic Cluster Expansion: body order, the density trick, completeness. |
| `10b_mace_theory.ipynb` | MACE: higher-order equivariant message passing = ACE + message passing; the design-space view. |
| `10c_mace_implementation.ipynb` | Block-by-block MACE in e3nn; training and evaluating forces/energies. |

### Part V — Applications (Lesson 11)

| Notebook | Topic |
|---|---|
| `11_molecular_dynamics_ase.ipynb` | Wrap a trained model as an ASE calculator; run MD; sanity checks (energy conservation, RDFs). |

## Repository Layout

```
e3nn_course/
├── README.md                 ← you are here
├── STYLE_GUIDE.md            ← authoring conventions for all notebooks
├── pyproject.toml / uv.lock  ← reproducible environment (uv)
├── course_utils/             ← shared helpers imported by every notebook
│   ├── equivariance.py       ← numerical equivariance test harness
│   ├── plotting.py           ← spherical harmonics / irreps / training visualizations
│   └── data.py               ← small datasets, neighbor lists, train/val splits
├── notebooks/                ← the lessons (Parts I–V)
└── papers/                   ← primary literature referenced throughout (see below)
```

## Primary References (in `papers/`)

**Core frameworks & foundational methods**

- Geiger & Smidt, *e3nn: Euclidean Neural Networks* (2022) — [Geiger arXiv 2207.09453 2022](papers/Geiger%20arXiv%202207.09453%202022.pdf)
- Thomas et al., *Tensor Field Networks* (2018) — [Thomas arXiv 1802.08219 2018](papers/Thomas%20arXiv%201802.08219%202018.pdf)
- Weiler et al., *3D Steerable CNNs* (NeurIPS 2018) — [Weiler NeurIPS 2018](papers/Weiler%20NeurIPS%202018.pdf)
- Battaglia et al., *Relational inductive biases and graph networks* (2018) — [Battaglia arXiv 1806.01261 2018](papers/Battaglia%20arXiv%201806.01261%202018.pdf)
- Fuchs et al., *SE(3)-Transformers* (NeurIPS 2020) — [Fuchs NeurIPS 2020](papers/Fuchs%20NeurIPS%202020.pdf)
- Anderson et al., *Cormorant* (NeurIPS 2019) — [Anderson NeurIPS 2019](papers/Anderson%20NeurIPS%202019.pdf)
- Liao & Smidt, *Equiformer* (ICLR 2023) — [Liao ICLR 2023](papers/Liao%20ICLR%202023.pdf)

**Equivariant interatomic potentials**

- Batzner et al., *NequIP: E(3)-equivariant GNNs for interatomic potentials* (Nat. Commun. 2022) — [Batzner Nat. Commun. 13 2453 2022](papers/Batzner%20Nat.%20Commun.%2013%202453%202022.pdf)
- Musaelian et al., *Allegro: local equivariant representations* (Nat. Commun. 2023) — [Musaelian Nat. Commun. 14 579 2023](papers/Musaelian%20Nat.%20Commun.%2014%20579%202023.pdf)
- Batatia et al., *MACE* (NeurIPS 2022) — [Batatia NeurIPS 2022](papers/Batatia%20NeurIPS%202022.pdf)
- Batatia et al., *The design space of E(3)-equivariant atom-centred potentials* — [Batatia arXiv 2205.06643 2022](papers/Batatia%20arXiv%202205.06643%202022.pdf) / [Batatia Nat. Mach. Intell. 7 56 2025](papers/Batatia%20Nat.%20Mach.%20Intell.%207%2056%202025.pdf)
- Kovács et al., *Evaluation of MACE* (2023) — [Kovacs J. Chem. Phys. 159 044118 2023](papers/Kovacs%20J.%20Chem.%20Phys.%20159%20044118%202023.pdf)
- Kovács et al., *MACE-OFF* — [Kovacs J. Am. Chem. Soc. 147 17598 2025](papers/Kovacs%20J.%20Am.%20Chem.%20Soc.%20147%2017598%202025.pdf)
- Batatia et al., *MACE-MP-0 foundation model* (2023) — [Batatia arXiv 2401.00096 2024](papers/Batatia%20arXiv%202401.00096%202024.pdf)

**Atomic Cluster Expansion & theory**

- Drautz, *Atomic cluster expansion* (PRB 2019) — [Drautz Phys. Rev. B 99 014104 2019](papers/Drautz%20Phys.%20Rev.%20B%2099%20014104%202019.pdf)
- Dusson et al., *ACE: completeness, efficiency, stability* (2022) — [Dusson J. Comput. Phys. 454 110946 2022](papers/Dusson%20J.%20Comput.%20Phys.%20454%20110946%202022.pdf)
- Nigam et al., *Unified theory of atom-centered representations and message passing* (JCP 2022) — [Nigam J. Chem. Phys. 156 204115 2022](papers/Nigam%20J.%20Chem.%20Phys.%20156%20204115%202022.pdf)
- Pozdnyakov & Ceriotti, *Incompleteness of GNNs for point clouds* — [Pozdnyakov arXiv 2201.07136 2022](papers/Pozdnyakov%20arXiv%202201.07136%202022.pdf)
- *Resolving the body-order paradox of MLIPs* (2026) — [Chong J. Chem. Phys. 164 064121 2026](papers/Chong%20J.%20Chem.%20Phys.%20164%20064121%202026.pdf)

**Long-range interactions & periodic systems**

- Grisafi & Ceriotti, *Incorporating long-range physics* (JCP 2019) — [Grisafi J. Chem. Phys. 151 204105 2019](papers/Grisafi%20J.%20Chem.%20Phys.%20151%20204105%202019.pdf)
- Kosmala et al., *Ewald-based message passing* (2023) — [Kosmala ICML 2023](papers/Kosmala%20ICML%202023.pdf)
- *Latent Ewald summation* — [Cheng npj Comput. Mater. 11 80 2025](papers/Cheng%20npj%20Comput.%20Mater.%2011%2080%202025.pdf)
- Kolafa & Perram, *Cutoff errors in Ewald summation* (1992) — [Kolafa Mol. Simul. 9 351 1992](papers/Kolafa%20Mol.%20Simul.%209%20351%201992.pdf)
- plus additional papers on crystals/periodicity, magnetic ordering, many-body expansions, and scaling (see `papers/`).

**External resource:** the official e3nn MRS Fall 2021 tutorial — <https://e3nn.org/e3nn-tutorial-mrs-fall-2021/>.
