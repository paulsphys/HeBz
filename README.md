This repository contains all the interaction potentials for Helium-Benzene interaction. This includes 

1) A Lennard-Jones potential
2) The analytical potential derived from DFT data given in [Lee 2003](https://doi.org/10.1063/1.1628217)
3) The analytical potential derived from CCSD(T) data given in [Shirkov 2024](https://doi.org/10.1021/acs.jpca.4c01491)
4) A gaussian process potential derived from CCSD(T)/CBS data.

The methods can be accessed as <code>LennardJones</code>, <code>Lee2003</code>, <code>Shirkov2024</code> and <code>V</code>. Each function takes $$x,y,z$$ coordinates 
in $$\AA$$ and output the potential $$K$$.
