import numpy as np
import random
import matplotlib.pyplot as plt
import os

class GeneticAlgorithmTSP:
    def __init__(self, cities, population_size, generations, mutation_rate, selection_method, init_method):
        self.cities = cities
        self.num_cities = len(cities)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.init_method = init_method
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        if self.init_method == "random":
            for _ in range(self.population_size):
                individual = list(np.random.permutation(self.num_cities))
                population.append(individual)
        elif self.init_method == "heuristic":
            for _ in range(self.population_size):
                individual = self.heuristic_individual()
                population.append(individual)
        elif self.init_method == "hybrid":
            for _ in range(self.population_size // 2):
                individual = list(np.random.permutation(self.num_cities))
                population.append(individual)
            for _ in range(self.population_size // 2):
                individual = self.heuristic_individual()
                population.append(individual)
        return population

    def heuristic_individual(self):
        # Implementa una heurística para inicializar el individuo
        individual = list(range(self.num_cities))
        random.shuffle(individual)
        return individual

    def fitness(self, individual):
        distance = 0
        for i in range(len(individual)):
            from_city = self.cities[individual[i]]
            to_city = self.cities[individual[(i+1) % len(individual)]]
            distance += np.linalg.norm(np.array(from_city) - np.array(to_city))
        return distance

    def select(self, population, fitnesses):
        tournament_size = 5
        tournament = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament]
        selected_index = tournament[np.argmin(tournament_fitnesses)]
        return population[selected_index]

    def crossover(self, parent1, parent2):
        start, end = sorted(random.sample(range(self.num_cities), 2))
        child = [None] * self.num_cities
        child[start:end] = parent1[start:end]
        pointer = end
        for gene in parent2:
            if gene not in child:
                if pointer >= self.num_cities:
                    pointer = 0
                child[pointer] = gene
                pointer += 1
        return child

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(self.num_cities), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    def run(self):
        best_individual = None
        best_fitness = float('inf')
        fitness_history = []
        
        for gen in range(self.generations):
            fitnesses = [self.fitness(ind) for ind in self.population]
            new_population = []
            for _ in range(self.population_size):
                parent1 = self.select(self.population, fitnesses)
                parent2 = self.select(self.population, fitnesses)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            
            self.population = new_population
            gen_best_fitness = min(fitnesses)
            gen_best_individual = self.population[fitnesses.index(gen_best_fitness)]
            fitness_history.append(gen_best_fitness)
            
            if gen_best_fitness < best_fitness:
                best_fitness = gen_best_fitness
                best_individual = gen_best_individual
            
            print(f"Generation {gen}: Best Fitness = {gen_best_fitness}")
        
        return best_individual, best_fitness, fitness_history

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_fitness(histories, output_dir):
    plt.figure(figsize=(10, 6))
    for fitness_history, method in histories:
        smoothed_fitness = smooth_curve(fitness_history)
        plt.plot(smoothed_fitness, label=method)
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.title('Fitness over Generations')
    plt.savefig(os.path.join(output_dir, 'fitness_comparison.png'))
    plt.show()

def plot_route(cities, individual, title, output_dir):
    x = [cities[i][0] for i in individual] + [cities[individual[0]][0]]
    y = [cities[i][1] for i in individual] + [cities[individual[0]][1]]
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png"))
    plt.show()

def main():
    output_dir = 'results_experiment2'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the cities as a list of coordinates
    cities = [(random.random(), random.random()) for _ in range(100)]
    
    methods = ["random", "heuristic", "hybrid"]
    histories = []
    best_individuals = {}
    
    for method in methods:
        print(f"Running Genetic Algorithm with {method} initialization")
        ga = GeneticAlgorithmTSP(cities, population_size=100, generations=3000, mutation_rate=0.01, selection_method="tournament", init_method=method)
        best_individual, best_fitness, fitness_history = ga.run()
        histories.append((fitness_history, method))
        best_individuals[method] = best_individual
        print(f"Best fitness ({method}): {best_fitness}")
        print(f"Best individual ({method}): {best_individual}")
    
    plot_fitness(histories, output_dir)
    
    for method, individual in best_individuals.items():
        plot_route(cities, individual, f"Best Route ({method})", output_dir)

if __name__ == "__main__":
    main()
