from operator import itemgetter
import numpy as np
from random import sample, random, gauss
from gym_reward import get_reward
from pytorch_model import get_random_state_dict
import sys
from copy import deepcopy


def mutate_funciton_individ(individ, mutation_rate=1):
    standatd_deviation = 2
    genes = individ['genes']  # state_dict for the model
    for key in genes.keys():  # for each layers's weight or bias list
        # for each neuron's weights to the others neurons it is connected to
        layer_parameters = genes[key]
        for i, neuron_parameter in enumerate(layer_parameters):
            if isinstance(neuron_parameter, list):  # "weights. each neuron have multiple connections"
                for j, connections_parameter in enumerate(neuron_parameter):
                    if (random() < mutation_rate):
                        genes[key][i][j] = gauss(connections_parameter, standatd_deviation)
            else:
                # each neuron have only one parameter in this group. Most probably a bias
                if (random() < mutation_rate):
                    genes[key][i] = gauss(neuron_parameter, standatd_deviation)

    individ['genes'] = genes
    individ['age'] = 0
    return individ


def mutate_subset(subset, mutate_funciton_individ):
    """
    Returns an array with mutated individuals from the subset
    :param subset:
    :param mutate_funciton_individ:
    :return:
    """
    new_mutated_subset = []
    for individ in subset:
        new_mutated_subset.append(mutate_funciton_individ(deepcopy(individ)))
    new_mutated_subset
    return new_mutated_subset


def create_initial_population(population_size, zeros=True):
    def create_genes():
        return get_random_state_dict()

    names = np.array(['fitness', 'genes', 'age'])
    initial_population = []
    for i in range(population_size):
        individ = {'fitness': 0, 'genes': create_genes(), 'age': 0}
        initial_population.append(individ)
    return initial_population


def get_individual_fitness(individual, render=False):
    '''
    Uses the gym envorment to get a fitness. Only uses direct encodeing from genes.
    :param individual:
    :return:
    '''
    # genes to phenotype, now direct mapping since genes is state dict and we only have one model
    fitness = get_reward(individual['genes'], render=render)
    individual['fitness'] = fitness
    return fitness


def give_fittness(population):
    for individual in population:
        individual['fitness'] = get_individual_fitness(individual)
        # print(individual['fitness'])


# population = [{fitness: , genes: , age: }

def simpleGA_A(initial_population, population_size, subset_size, mutate_function, number_of_generations,
               generation_replacment):
    population = initial_population
    N = population_size
    F = give_fittness(initial_population)
    for g in range(number_of_generations):
        print('Generation ' + str(g))
        # subset_S = np.random.choice(initial_population, subset_size)
        # create new samples
        subset_S = sample(population, subset_size)
        mutated_individuals = mutate_subset(subset_S, mutate_function)
        give_fittness(mutated_individuals)
        # combine samples with population
        mutated_individuals.sort(key=itemgetter('fitness'), reverse=True)
        population = population + mutated_individuals[:generation_replacment]

        # remove the oldest
        population.sort(key=itemgetter('age'), reverse=False)
        population = population[:-(generation_replacment)]

        # get_individual_fitness(population[0],render=True)
        # print(subset_S)
        # increase age and print out stuff
        population.sort(key=itemgetter('fitness'), reverse=True)
        print('Elite')
        for rank, individ in enumerate(population):
            print("rank: {}  Fitness: {}  Age: {} ".format(str(rank), str(individ['fitness']), str(individ['age'])))
            individ['age'] += 1

    print(population.sort(key=itemgetter('fitness'), reverse=True))


if __name__ == '__main__':
    population_size = 50
    subset_size = 40
    replacement_per_generation = 1
    generations = 30
    initial_population = create_initial_population(population_size)
    simpleGA_A(initial_population, population_size, subset_size, mutate_funciton_individ, generations,
               replacement_per_generation)
