from .all_terms import run as run_all_terms
from .no_descriptor_terms import run as run_no_descriptor_terms
from .no_energy_terms import run as run_no_energy_terms
from .no_geometric_terms import run as run_no_geometric_terms
from .benchmark import run as run_benchmark
from argparse import ArgumentParser


algorithms = {
    "all": run_all_terms,
    "no_desc": run_no_descriptor_terms,
    "no_eng": run_no_energy_terms,
    "no_geo": run_no_geometric_terms,
    "bench": run_benchmark
}
parser = ArgumentParser(prog="EDA4PD", 
                        description="An Estimation of Distribution Algorithm "
                                    "for Protein Design.")
parser.add_argument("-t", 
                    "--fitness_terms", 
                    type=str, 
                    choices=list(algorithms.keys()),
                    help="Indicates which fitness function terms to turn off.")
parser.add_argument("-n",
                    "--max_generations",
                    type=int,
                    default=1000,
                    help="Indicates for how many generations to run EDA4PD.")
parser.add_argument("-p",
                    "--population_size",
                    type=int,
                    default=5,
                    help="The number of probability distributions in "
                         "the population.")
parser.add_argument("-s",
                    "--sample_size",
                    type=int,
                    default=3,
                    help="The number of sequences to sample from each "
                         "probability distribution in the population.")
args = parser.parse_args()
run_algorithm = algorithms[args.fitness_terms]
run_algorithm(args.max_generations, 
              args.population_size, 
              args.sample_size)
