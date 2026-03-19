# Spectral-HAC Replication Package

This repository contains the replication materials for the paper:

**“Bulk–Boundary Decomposition for Adaptive Windows in Spectral/HAC Estimation: Mechanism, Scale, and Certified Negativity.”**

## Overview

The repository provides the code and output files used to reproduce the main Monte Carlo results reported in the paper. The experiments study adaptive lag windows in HAC estimation, with special attention to:

- finite-sample PSD properties,
- structural differences between hard-cutoff and Bartlett-type constructions,
- failure frequencies,
- inferential consequences,
- repair procedures,
- and multivariate extensions.

## Repository structure

```text
spectral-hac-replication/
├── code/        # Python scripts used to run the simulations
├── data/        # Input data or saved intermediate objects (if applicable)
├── figures/     # Figures reported in the paper
├── tables/      # LaTeX tables reported in the paper
└── README.md
