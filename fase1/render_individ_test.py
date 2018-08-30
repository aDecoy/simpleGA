from algorithms import get_individual_fitness
import pickle

individ_file='beste_individ.pickle'
with open(individ_file, 'rb') as f:
    individual= pickle.load(f)

    print(get_individual_fitness(individual, number_off_trials=80, render=True))



