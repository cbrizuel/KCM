# KCM

This repository contains the source code for the evolutionary algorithm used in the article titled "An Evolutionary Algorithm for a Key-Cutting-Machine Approach in the Design of Proteins". This source code is provided mainly for the purpose of reproducing the results presented in said article.

## Requirements

The KCM program requires **Python version 3.7** and has the following package dependencies:

- [NumPy](https://numpy.org/install/) v1.21.
- [BioPython](https://biopython.org/wiki/Packages) v1.79.
- [Pandas](https://pypi.org/project/pandas/) v1.3.5.
- [Matplotlib](https://matplotlib.org/stable/users/getting_started/index.html#installation-quick-start) v3.5.3.
- [SciPy](https://scipy.org/install/#pip-install) v1.7.1.
- [Hugging Face's Transformers](https://github.com/huggingface/transformers?tab=readme-ov-file#with-conda) v4.28.1.

The KCM program also utilizes the following external programs:

- [ESM2 (`esm2_t33_650M_UR50D`) and ESMFold (`esmfold_v1`)](https://github.com/facebookresearch/esm?tab=readme-ov-file#repostart).
- [PyRosetta](https://www.pyrosetta.org/downloads#h.c0px19b8kvuw) v2020.10.

The KCM program was tested in a Linux machine with the following characteristics:

- **OS**: Pop!_OS 22.04 LTS
- **CPU**: Intel Core i5-9400 (6 cores @ 2.90 GHz)
- **RAM**: 32 GB DDR4
- **GPU**: NVIDIA GeForce RTX 3090 Ti (24 GB GDDR6X)
- **CUDA SDK**: 11.8.20220929
- **NVIDIA Driver**: 520.61.05
- **Linux kernel version**: 6.6.10-76060610-generic x86_64
- **Python version**: 3.7.12

## Installation

All package dependencies can be installed individually using Anaconda. Before installing these dependencies, modify the `~/.condarc` file to add the following channels:

**For WEST coast**:
```
channels: 
    - https://conda.rosettacommons.org
```

**For EAST coast**:
```
channels:
    - https://conda.graylab.jhu.edu
```

After adding this channel, run the following commands to install the required packages; notice that a new Anaconda environment, called `KCM`, will be created:

```bash
$ conda create --name KCM python=3.7.13
$ conda activate KCM
$ conda install numpy=1.21
$ conda install scipy=1.7.1
$ conda install -c conda-forge biopython=1.79
$ conda install -c conda-forge pandas=1.3.5
$ conda install -c conda-forge matplotlib=3.5.3
$ conda install conda-forge::transformers=4.28.1
$ conda install pyrosetta=2020.10
```

Installing these packages should take about 25 minutes.

### Using a local ESMFold installation with KCM

KCM uses [ESMFold](https://esmatlas.com/resources?action=fold) to predict the folding structure of each designed amino acid sequence. By default, KCM will use the [official API](https://esmatlas.com/about#api) to request the prediction through the Internet. To install ESMFold locally, we recommend following the [official installation instructions](https://github.com/facebookresearch/esm?tab=readme-ov-file#getting-started-with-this-repo-) for installing the `esmfold` Anaconda environment. 

Once this environment is installed, we recommend runnig ESMFold in a local server using [Gunicorn](https://gunicorn.org/) and providing its URL to KCM by modifying the `esmfold_url` value in the `config.py` file. The local server must receive as input a string containing an amino acid sequence, and must return as output a string containing the raw PDB file of the predicted structure.

## Running KCM

To run the KCM program, simply use the following commands:

```bash
$ conda activate KCM
$ python -m <KCM repository> -t <fitness_terms>
```

where `<fitness_terms>` indicates what kind of terms should be considered when calculating the fitness values of the designed proteins. There are five different options for `<fitness_terms>`, which are: `all`, `no_desc`, `no_eng`, `no_geo`, and `bench`.

The fitness function in KCM consists of a combination of terms measuring three different aspects of each designed protein: geometrical similarity of the protein backbone with the target structure, Rosetta energy score, and iLearn and ESM-2 descriptor values. These terms can be turned off individually using the `-t <fitness_terms>` argument shown above. For example, using the command `python -m KCM -t no_desc` will run KCM with the iLearn and ESM-2 descriptors turned off, while the command `python -m KCM -t all` will run the program with all terms turned on.

In addition to `-t <fitness_terms>`, KCM accepts the following arguments for customizing other parts of the algorithm: 

- `-n <number>` or `--max_generations <number>`: maximum number of generations. The default value is `1000` (one thousand).
- `-p <number>` or `--population_size <number>`: population size. In KCM, each individual in the population is a probability distribution. The default value is `1`.
- `-s <number>` or `--sample_size <number>`: sample size; i.e., the number of sequences to sample from each distribution in the population. The default value is `1`.
- `-i <folder name>` or `--input_folder <folder name>`: the name of the local folder containing the PDB files to be used by KCM as design targets. The default value is `demo_pdbs`. Other valid values are `target_pdbs` and `benchmark_pdbs`.

### Running a demo

You can check the installation by running a small demo using the following command:

```bash
$ python -m <KCM repository> -i demo_pdbs -t no_desc -n 1 -p 1
```

This command will run KCM on a small protein for a single generation. The demo should take about 1 hour to finish if the ESMFold API is used instead of a local installation. During execution, a file named `1y32_datos_guardados0.txt` will be created in the current directory. This file is created for each generation of the KCM evolutionary algorithm (in this case, only one generation is executed, so only one file is created), and describes the solutions found by each island (numbered 1 to 20). Relevant values in this file are the following:

- `best_execution`: the amino acid sequences sampled from each probability distribution in the algorithm's population.
- `best_execution_dist`: the amino acid probability distributions in the algorithm's population.
- `pdb_select`: the raw PDB prediction made by ESMFold for the sequence found by the algorithm with the best fitness value over-all. 

### Running the benchmark as described in the article

To run KCM using the 23 CATH proteins used as benchmark in the original article, use the following command:

```bash
$ python -m <KCM repository> -i benchmark_pdbs -t bench
```

### Running the IDR-2009 design described in the article

To run KCM over the IDR-2009 peptide as described in the original article, you need to run the following commands separetely:

```bash
$ python -m <KCM repository> -i target_pdbs -t all
$ python -m <KCM repository> -i target_pdbs -t no_desc
$ python -m <KCM repository> -i target_pdbs -t no_geo
$ python -m <KCM repository> -i target_pdbs -t no_eng
```
