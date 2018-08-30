from operator import itemgetter
import numpy as np
from random import sample, random, gauss, choices
from gym_reward import save_checkpoint,load_new_model_state,resume_from_checkpoint,get_reward
from pytorch_model import get_random_state_dict
import sys
from copy import deepcopy
import pickle

best_individual_file_name='beste_individ.pickle'

def mutate_funciton_individ(individ, mutation_rate=0.1,standatd_deviation = 3):

    genes = individ['genes']  # state_dict for the model
    for key in genes.keys():  # for each layers's weight or bias list
        # for each neuron's weights to the others neurons it is connected to
        layer_parameters = genes[key]
        for i, neuron_parameter in enumerate(layer_parameters):
            if isinstance(neuron_parameter, list):  # "weights. each neuron have multiple connections"
                for j, connections_parameter in enumerate(neuron_parameter):
                    if (random() < mutation_rate):
                        # genes[key][i][j] = gauss(connections_parameter, standatd_deviation)
                        genes[key][i][j] = random()*2-1
            else:
                # each neuron have only one parameter in this group. Most probably a bias
                if (random() < mutation_rate):
                    # genes[key][i] = gauss(neuron_parameter, standatd_deviation)
                    genes[key][i] = random()*2-1

    individ['genes'] = genes
    individ['age'] = 0
    return individ


def mutate_subset(subset, mutate_funciton_individ, number_of_individs_to_mutate=None, offspring_from_each_individ=1):
    """
    Returns an array with mutated individuals from the number_of_individs_to_mutate best individuals in the subset.
    :param subset:
    :param mutate_funciton_individ:
    :param number_of_individs_to_mutate:
    :return:
    """
    if number_of_individs_to_mutate is None:
        number_of_individs_to_mutate = len(subset)
    new_mutated_subset = []
    subset.sort(key=itemgetter('fitness'), reverse=True)
    for i in range(number_of_individs_to_mutate):
        for _ in range(offspring_from_each_individ):
            new_mutated_subset.append(mutate_funciton_individ(deepcopy(subset[i])))
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


def get_individual_fitness(individual, number_off_trials=10, render=False, ):
    '''
    Uses the gym envorment to get a fitness. Only uses direct encodeing from genes.
    :param individual:
    :return:
    '''
    # genes to phenotype, now direct mapping since genes is state dict and we only have one model
    fitness_observations = []
    fitness_observations.append(get_reward(individual['genes'], render=render))
    for i in range(number_off_trials - 1):
        fitness_observations.append(get_reward(individual['genes'], render=render))
    fitness = sum(fitness_observations) / float(len(fitness_observations))
    individual['fitness'] = fitness
    return fitness


def give_fittness(population, number_off_trials=10):
    """

    :param population:
    :param number_off_trials:
    :return:
    """
    for individual in population:
        individual['fitness'] = get_individual_fitness(individual, number_off_trials, render=False)
        # print(individual['fitness'])


# population = [{fitness: , genes: , age: }

def simpleGA_A(initial_population, population_size, subset_size, mutate_function, number_of_generations,
               generation_replacment=1, offspring_from_each_indivual=1):
    """
    Regularized evolution for image classifier architecture search
    :param initial_population:
    :param population_size:
    :param subset_size:
    :param mutate_function:
    :param number_of_generations:
    :param generation_replacment:
    :return:
    """
    population = initial_population
    give_fittness(population)
    highest_fitness_each_generation = []

    for g in range(number_of_generations):
        print('Generation ' + str(g))
        # subset_S = np.random.choice(initial_population, subset_size)
        # create new samples
        subset_S = sample(population, subset_size)
        mutated_individuals = mutate_subset(subset_S, mutate_function, generation_replacment,
                                            offspring_from_each_indivual)

        # Here you are suppose to train mutated_individuals, but we can not really do that for all tasks.

        give_fittness(mutated_individuals)
        # combine samples with population
        population = population + mutated_individuals

        # remove the oldest
        population.sort(key=itemgetter('age'), reverse=False)
        population = population[:-(generation_replacment)]
        # print(subset_S)
        # increase age and print out stuff
        population.sort(key=itemgetter('fitness'), reverse=True)
        print('Elite')
        for rank, individ in enumerate(population):
            print("rank: {}  Fitness: {}  Age: {} ".format(str(rank), str(individ['fitness']), str(individ['age'])))
            individ['age'] += 1
            # get_individual_fitness(population[0],render=True)
            highest_fitness_each_generation.append(population[0]['fitness'])

    print(population.sort(key=itemgetter('fitness'), reverse=True))


def simpleGA_B(initial_population, population_size, subset_size, mutate_function, number_of_generations,
               generation_replacment):
    population = initial_population
    give_fittness(population)
    highest_fitness_each_generation = []

    population.sort(key=itemgetter('fitness'), reverse=True)
    for g in range(number_of_generations):
        subset_S = population[:subset_size]
        mutated_individuals = mutate_subset(subset_S, mutate_function, generation_replacment)
        give_fittness(mutated_individuals)

        population = population[:-(generation_replacment)]
        population = population + mutated_individuals
        population.sort(key=itemgetter('fitness'), reverse=True)
        print('Generation ' + str(g))
        for rank, individ in enumerate(population):
            print("rank: {}  Fitness: {}  Age: {} ".format(str(rank), str(individ['fitness']), str(individ['age'])))
            individ['age'] += 1
            highest_fitness_each_generation.append(population[0]['fitness'])


def simpleGA_C(initial_population, population_size, mutate_function, number_of_generations, subset_size=None,
               number_of_elites=5, offspring_from_each_indivual=1, number_of_individs_to_mutate=None, number_of_trials=20):
    """
    DeepNeuroevolution: GeneticAlgorithmsareaCompetitiveAlternativefor TrainingDeepNeuralNetworksforReinforcementLearning
FelipePetroskiSuch VashishtMadhavan EdoardoConti JoelLehman KennethO.Stanley JeffClune
    :param initial_population:
    :param population_size:
    :param subset_size:
    :param mutate_function:
    :param number_of_generations:
    :param generation_replacment:
    :param number_of_elites:
    :return:
    """
    if subset_size is None:
        subset_size = population_size
    population = initial_population
    # if number_of_individs_to_mutate=None
    #     number_of_individs_to_mutate= population_size

    print('Giving fittnes to initial population')
    give_fittness(population, number_off_trials=5)

    highest_fitness_each_generation = []
    subset_S = choices(population, k=subset_size)
    mutated_individuals = mutate_subset(subset_S, mutate_function,
                                        number_of_individs_to_mutate=subset_size,
                                        offspring_from_each_individ=offspring_from_each_indivual)
    give_fittness(mutated_individuals, number_of_trials)
    mutated_individuals.sort(key=itemgetter('fitness'), reverse=True)
    elite_set = mutated_individuals[:number_of_elites]

    population_without_elite = []
    for individ in mutated_individuals:
        if individ not in elite_set:
            population_without_elite.append(individ)
    population = elite_set + population_without_elite


    for rank, individ in enumerate(population):
        print("rank: {}  Fitness: {}  Age: {} ".format(str(rank), str(individ['fitness']), str(individ['age'])))
        individ['age'] += 1
        highest_fitness_each_generation.append(population[0]['fitness'])

    for g in range(number_of_generations):
        #create new generation
        subset_S = choices(population, k=subset_size)
        mutated_individuals = mutate_subset(subset_S, mutate_function,
                                            number_of_individs_to_mutate=number_of_individs_to_mutate,
                                            offspring_from_each_individ=offspring_from_each_indivual)
        give_fittness(mutated_individuals, number_of_trials)
        #set elite
        mutated_individuals.sort(key=itemgetter('fitness'), reverse=True)
        elite_set_canditates = elite_set + mutated_individuals[:number_of_elites]
        give_fittness(elite_set_canditates, number_off_trials=20)
        elite_set_canditates.sort(key=itemgetter('fitness'), reverse=True)
        elite_set = elite_set_canditates[:number_of_elites]

        #combine into one population set
        population_without_elite = []
        for individ in mutated_individuals:
            if individ not in elite_set:
                population_without_elite.append(individ)

        print('New elites found: {}'.format(population_size-len(population_without_elite)))
        population = elite_set + population_without_elite
        population = population[:population_size]
        population.sort(key=itemgetter('fitness'), reverse=True)
        print('Generation ' + str(g))
        for rank, individ in enumerate(population):
            print("rank: {}  Fitness: {}  Age: {} ".format(str(rank), str(individ['fitness']), str(individ['age'])))
            individ['age'] += 1
            highest_fitness_each_generation.append(population[0]['fitness'])

        if g%5 == 0 :
            with open(best_individual_file_name, 'wb') as f:
                pickle.dump(population[0],f)
                print('Saved best individual')

    print(highest_fitness_each_generation)
    with open('highest_fintess.txt', 'a') as f:
        for item in highest_fitness_each_generation:
            f.write(" %s," % item)
        f.write("\n" % item)

    get_individual_fitness(population[0],number_of_trials,render=True)


if __name__ == '__main__':
    population_size = 20
    subset_size = 10
    replacement_per_generation = 4
    generations = 60
    initial_population = create_initial_population(population_size)
    # simpleGA_A(initial_population, population_size, subset_size, mutate_funciton_individ, generations, replacement_per_generation)
    # simpleGA_B(initial_population, population_size, subset_size, mutate_funciton_individ, generations, replacement_per_generation)

    simpleGA_C(initial_population, population_size, mutate_funciton_individ, generations,number_of_trials=8,number_of_elites=5)
