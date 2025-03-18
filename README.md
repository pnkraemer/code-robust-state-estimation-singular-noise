# JAX code: Numerically robust Gaussian state estimation with singular observation noise

This repository contains the JAX implementation of the experiments from the paper:

> Krämer, Nicholas, and Filip Tronarp. "Numerically robust Gaussian state estimation with singular observation noise." arXiv preprint arXiv:2503.10279 (2025).

[Here](https://arxiv.org/abs/2503.10279) is a link.

Use the following bibtex to cite the paper:

```bibtex 
@article{kramer2025numerically,
  title={Numerically robust Gaussian state estimation with singular observation noise},
  author={Kr{\"a}mer, Nicholas and Tronarp, Filip},
  journal={arXiv preprint arXiv:2503.10279},
  year={2025}
}
```

## Installation
Ensure you have JAX installed correctly for your hardware (CPU/GPU/TPU). 
Refer to the [JAX installation guide](https://github.com/google/jax#installation) if needed.

Install the required dependencies and the source:

```commandline
pip install .
```

To ensure that everything works correctly, run the tests:

```commandline
pytest
```
## Running Experiments

The experiments are in the `experiments/` directory. 

