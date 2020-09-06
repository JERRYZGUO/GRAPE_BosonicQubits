# GRAPE_BosonicQubits

Repository for simulating Hubbard model with multiple oscillator-qubit systems using GRAPE algorithm.

## Installation

Fork the repo, and follow the instructions [here](http://qutip.org/docs/latest/installation.html) to install qutip devloper version from the qutip folder.

## Project Structure

Most of the codes are given in this repo except some not essential ones, which can be available upon request.
* Within folder **code**, **dynamics.py** is the main file to run. **config.json** stores the parameters of the physical system and GRAPE algorithm. **optimizedPulse** stores information of two sample optimized pulse amplitudes.

* Folder **qutip** is mostly code from the released qutip codes from [here](https://github.com/qutip/qutip/tree/master/qutip). The major modifications are in **qutip/control/fidcomp.py**, where two new classes *fidCompModified* and *fidCompVariational* are created for dynamics simulation and variational method respectively.
