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


if __name__ == '__main__':
    population = [0.81, 0.56, 0.77, 0.63, 0.42, 0.99, 0.65, 0.28, 0.47, 0.84, 0.59, 0.73, 0.36, 0.92, 0.21, 0.69, 0.58, 0.33, 0.97, 0.48]
    print(SelectionMethods.elite(population, 30))
