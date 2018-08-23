from operator import itemgetter
import numpy as np
from random import sample, random, gauss
from gym_reward import get_reward
from pytorch_model import get_random_state_dict
import sys

def mutate_funciton_individ(individ, mutation_rate=0.2):
    standatd_deviation = 2
    genes = individ['genes']  # state_dict for the model
    for key in genes.keys(): #for each layers's weight or bias list
        # for each neuron's weights to the others neurons it is connected to
        layer_parameters = genes[key]
        for i, neuron_parameter in enumerate(layer_parameters):
            if isinstance(neuron_parameter,list): #"weights. each neuron have multiple connections"
                for j, connections_parameter in enumerate(neuron_parameter):
                    if (random() < mutation_rate):
                        genes[key][i][j] = gauss(connections_parameter, standatd_deviation)
            else:
                # each neuron have only one parameter in this group. Most probably a bias
                if (random() < mutation_rate):
                    genes[key][i] = gauss(neuron_parameter, standatd_deviation)

    individ['genes'] = genes
    return individ


def mutate_subset(subset, mutate_funciton_individ):
    for individ in subset:
        mutate_funciton_individ(individ)
    return subset


def create_initial_population(population_size, zeros=True):
    def create_genes():
        return get_random_state_dict()

    names = np.array(['fitness', 'genes', 'age'])
    initial_population = []
    for i in range(population_size):
        individ = {'fitness': 0, 'genes': create_genes(), 'age': 0}
        initial_population.append(individ)
    return initial_population


def get_individual_fitness(individual):
    '''
    Uses the gym envorment to get a fitness. Only uses direct encodeing from genes.
    :param individual:
    :return:
    '''
    #genes to phenotype, now direct mapping since genes is state dict and we only have one model
    fitness=get_reward(individual['genes'],render=False)
    individual['fitness']=fitness
    return fitness


def give_fittness(population):
    for individual in population:
        individual['fitness'] = get_individual_fitness(individual)


# population = [{fitness: , genes: , age: }

def simpleGA_A(initial_population, population_size, subset_size, mutate_function, number_of_generations,
               generation_replacment):
    population = initial_population
    N = population_size
    F = give_fittness(initial_population)
    for g in range(number_of_generations):
        # subset_S = np.random.choice(initial_population, subset_size)
        subset_S = sample(population, subset_size)
        mutate_subset(subset_S, mutate_function)
        give_fittness(subset_S)
        subset_S.sort(key=itemgetter('fitness'), reverse=True)
        population.sort(key=itemgetter('age'), reverse=True)
        population = population[:-generation_replacment]
        population = population + subset_S[:generation_replacment]
        # print(subset_S)

    print(population.sort(key=itemgetter('fitness'), reverse=True))


if __name__ == '__main__':
    population_size = 20
    subset_size = 6
    replacement_per_generation=1
    generations= 10
    initial_population = create_initial_population(population_size)
    simpleGA_A(initial_population, population_size, subset_size, mutate_funciton_individ, generations, replacement_per_generation)
