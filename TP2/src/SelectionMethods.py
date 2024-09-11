import numpy as np


class SelectionMethods:
    @staticmethod
    def elite(population: list[float], K: int) -> list[float]:
        """
        Selects the individuals from the population based on their scores
        """
        population_lenght = len(population)

        population.sort(reverse=True)

        if K <= population_lenght:
            return population[0:K]

        population_with_repetition = []
        for i in range(K):
            population_with_repetition.append(population[i % population_lenght])

        return population_with_repetition

    @staticmethod
    def roulette(population: list[float], K: int) -> list[float]:
        """
         Selects individuals from the population based on their scores in a stochastic manner.
        """
        total_fitness = sum(population)
        relative_fitness = [score / total_fitness for score in population]

        cumulative_fitness = []
        cumulative_sum = 0.0
        for fitness in relative_fitness:
            cumulative_sum += fitness
            cumulative_fitness.append(cumulative_sum)

        random_numbers = np.random.uniform(0, 1, K)

        selected_population = []
        for random_number in random_numbers:
            for i in range(1, len(cumulative_fitness) + 1):
                if cumulative_fitness[i-1] < random_number <= cumulative_fitness[i]:
                    selected_population.append(population[i])
                    break

        print(total_fitness)
        print(relative_fitness)
        print(cumulative_fitness)
        print(random_numbers)
        print(selected_population)

        return selected_population


if __name__ == '__main__':
    population1 = [0.81, 0.56, 0.77, 0.63, 0.42, 0.99, 0.65, 0.28, 0.47, 0.84, 0.59, 0.73, 0.36, 0.92, 0.21, 0.69, 0.58, 0.33, 0.97, 0.48]
    population2 = [81, 56, 77, 63, 42, 99, 65, 28, 47, 84, 59, 73, 36, 92, 21, 69, 58, 33, 97, 48]
    population3 = [3, 6, 11, 14, 1]
    #print(SelectionMethods.elite(population, 30))
    SelectionMethods.roulette(population3, 3)
