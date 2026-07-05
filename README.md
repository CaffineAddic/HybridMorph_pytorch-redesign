# HybridMorph (PyTorch Redesign)

> Experimental redesign of the HybridMorph framework with a modular PyTorch implementation.

[![Status](https://img.shields.io/badge/Status-Archived%20Research-yellow)](#)
[![Python](https://img.shields.io/badge/Python-3-blue)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-red)](#)

> **Project Status**
>
> This repository is **unfinished**. Development was paused because several key medical imaging components available in the original TensorFlow implementation (through the VoxelMorph ecosystem) did not yet have mature or feature-complete PyTorch equivalents at the time of development.
>
> Rather than producing an incomplete or incompatible implementation, the project was archived while the research was completed using the original framework.

---

## Overview

This repository was intended to be a clean PyTorch redesign of the original HybridMorph research implementation. The objective was to reorganize the codebase into a modular, extensible architecture while preserving the methodology introduced in the HybridMorph paper.

Unlike the original repository, which contains the implementation used in the published research, this project focuses on software engineering and maintainability.

---

## Motivation

The original HybridMorph implementation evolved rapidly during experimentation and publication.

This redesign aimed to:

- migrate the framework fully to PyTorch
- modularize the codebase
- simplify experimentation
- improve reproducibility
- provide a foundation for future development

---

## Why development stopped

At the time this repository was developed, parts of the medical image registration ecosystem available to TensorFlow (particularly components used by VoxelMorph) were not yet available—or sufficiently mature—in PyTorch.

As a result, reproducing the full functionality of the original framework would have required reimplementing significant portions of the underlying infrastructure.

Rather than maintaining a partially functional implementation, development was paused after establishing the project structure and core components.

---

## Current Status

Implemented:

- Project restructuring
- Modular architecture
- Initial model migration
- Training pipeline

Not completed:

- Full registration pipeline
- Feature parity with the original HybridMorph implementation
- Reproduction of published experiments

---

## Related Repository

For the complete implementation accompanying the publication, see:

**HybridMorph**

*HybridMorph: Bridging the Gap between Synthetic and Real Data for Accurate MR Image Registration.*

---

## Future Work

This repository could be revived now that the PyTorch medical imaging ecosystem has matured substantially, particularly with projects such as:

- MONAI
- TorchIO
- Kornia
- Updated PyTorch registration libraries

---

## Author

**Saumya Roy**

Portfolio: https://caffineaddic.github.io
