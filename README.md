[![DOI](https://zenodo.org/badge/999627159.svg)](https://doi.org/10.5281/zenodo.17982887)

This repository contains all the interaction potentials for Helium-Benzene interaction. This includes 

1) A Lennard-Jones potential
2) The analytical potential derived from DFT data given in [Lee 2003](https://doi.org/10.1063/1.1628217)
3) The analytical potential derived from CCSD(T) data given in [Shirkov 2024](https://doi.org/10.1021/acs.jpca.4c01491)
4) A gaussian process potential derived from CCSD(T)/CBS data jointly developed by the [Del Maestro Group](https://delmaestro.org/adrian/) and the [Vogiatzis Group](https://vogiatzis.utk.edu/) at UTK.

The methods can be accessed as <code>LennardJones</code>, <code>Lee2003</code>, <code>Shirkov2024</code> and <code>V</code>. Each function takes $$x,y,z$$ coordinates 
in Å and outputs the potential in Kelvin.

## Installation
For now, we have not uploaded to pypi but the scripts can be installed directly
from git:
```console
pip install git+https://github.com/paulsphys/HeBz.git
```
