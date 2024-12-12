# EDA4PD

This repository contains the source code for the evolutionary algorithm used in the article titled "An Evolutionary Algorithm for a Key-Cutting-Machine Approach in the Design of Proteins". This source code is provided mainly for the purpose of reproducing the results presented in said article.

This program was tested in a Linux machine with the following characteristics:

- **OS**: Pop!_OS 22.04 LTS
- **CPU**: Intel Core i5-9400 (6 cores @ 2.90 GHz)
- **RAM**: 32 GB DDR4
- **GPU**: NVIDIA GeForce RTX 3090 Ti (24 GB GDDR6X)
- **CUDA SDK**: 11.8.20220929
- **NVIDIA Driver**: 520.61.05

### Installation

Software dependencies for EDA4PD can be installed using [Anaconda](https://docs.anaconda.com/anaconda/install/#).
In Linux, this can be achieved by running the following command:

```
$ conda env create -f <path to the EDA4PD repo>/environment.yml -n <custom_name>
```

Remember to activate the environment afterwards:

```
$ conda activate <custom_name>
```

### Running EDA4PD

To run the EDA4PD program in Linux, simply use the following command:

```
$ python -m EDA4PD -t <fitness_terms>
```

where `<fitness_terms>` indicates what kind of terms should be considered when calculating the fitness values of the designed proteins. There are five different options for `<fitness_terms>`, which are: `all`, `no_desc`, `no_eng`, `no_geo`, and `bench`.

The fitness function in EDA4PD consists of a combination of terms measuring three different aspects of each designed protein: geometrical similarity of the protein backbone with the target structure, Rosetta energy score, and iLearn and ESM-2 descriptor values. These terms can be turned off individually using the `-t <fitness_terms>` argument showed above. For example, using the command `python -m EDA4PD -t no_desc` will run EDA4PD with the iLearn and ESM-2 descriptors turned off, while the command `python -m EDA4PD -t all` will run the program with all terms turned on.

### Customization

The input for EDA4PD is a PDB file, which must be stored in the `target_pdb` folder. This folder currently holds the PDB file (named `cesarp.pdb`) for the IDR-2009 peptide. 
If you wish to run EDA4PD with a different target protein, simply store the PDB file of your protein in said folder (you can delete the `cesarp.pdb` file).

In addition to `-t <fitness_terms>`, EDA4PD take the following arguments for customizing other parts of the algorithm: 

- `-n <number>` or `--max_generations <number>`: maximum number of generations. Default value is `1000` (one thousand).
- `-p <number>` or `--population_size <number>`: population size. In EDA4PD, each individual in the population is a probability distribution. Default value is `5`.
- `-s <number>` or `--sample_size <number>`: sample size; i.e., the number of sequences to sample from each distribution in the population. Default value is `3`.

##### Using a local ESMFold installation with EDA4PD

EDA4PD uses [ESMFold](https://esmatlas.com/resources?action=fold) to predict the folding structure of each designed amino acid sequence. By default, EDA4PD will use the [official REST API](https://esmatlas.com/about#api) to request the prediction remotely. To use instead a local installation of ESMFold, we suggest running a local server using [Gunicorn](https://gunicorn.org/) and providing its URL to EDA4PD by modifying the `esmfold_url` value in the `config.py` file. The local server must recieve as input a string containing an amino acid sequence, and must return as output a string containing the raw contents of the PDB file of the predicted structure. The local server's input and output must _not_ be JSON encoded. 

### Reproducing the benchmark in the paper

To run EDA4PD using the 23 CATH proteins used as benchmark in the paper, use the following command:

```
$ python -m EDA4PD -t bench
```
