import math
import scipy.stats as stats

import numpy as np
from time import perf_counter
from scipy.stats import beta as b



class PriceSetter2:
    def __init__(self, rounds, alpha, beta):
        """
        Initialize the price setter.
        In this settings, the values of the costumers is distributed according to a beta distribution with the given parameters alpha and beta.

        Args:
            rounds (int): the number of rounds to simulate
            alpha (float): the alpha parameter of the beta distribution, bigger than 0
            beta (float): the beta parameter of the beta distribution, bigger than 0
        """
        self.alpha = alpha
        self.beta = beta
        self.last_proposition = None

        # Créer une liste d'index allant de 0 à 1 avec un pas de 0.001
        indexes = np.arange(0.001, 1.001, 0.001)

        # Calculer les valeurs de l'index multipliées par la probabilité
        table = []
        for idx in indexes:
            prob_sup = 1 - b.cdf(idx, self.alpha, self.beta)
            valeur = idx * prob_sup
            table.append([idx, valeur])

        self.table = table

        self.opt_index = max(self.table, key=lambda x: x[1])[0]

        # Affichage de la liste
        #for row in tableau:
            #print(float(row[0]), float(row[1]))

    def set_price(self, t):
        """
        Return the price at time t.

        Args:
            t (int): the time period

        Returns:
            float: the price at time t
        """

        return self.opt_index

    def update(self, t, outcome):
        """
        Update the price setter based on the outcome of the previous period.

        Args:
            t (int): the time period
            outcome (int): the outcome of the previous period - true if the product was sold, false otherwise
        """





def simulate(simulations, rounds, alpha, beta):
    simulations_results = []
    for _ in range(simulations):
        start = perf_counter()
        price_setter = PriceSetter2(rounds, alpha, beta)
        end = perf_counter()
        if end - start > 3:
            raise Exception("The initialization of the price setter is too slow.")
        revenue = 0

        for t in range(rounds):
            customer_value = np.random.beta(alpha, beta)
            start = perf_counter()
            price = price_setter.set_price(t)
            end = perf_counter()
            if end - start > 0.1:
                raise Exception("The set_price method is too slow.")

            if customer_value >= price:
                revenue += price

            start = perf_counter()
            price_setter.update(t, customer_value >= price)
            end = perf_counter()
            if end - start > 0.1:
                raise Exception("The update method is too slow.")

        simulations_results.append(revenue)

    return np.mean(simulations_results)


ALPHA_BETA_VALUES = [(2, 2), (4, 2), (2, 4), (4, 4), (8, 2), (2, 8), (8, 8)]
THRESHOLDS = [258, 409, 158, 284, 571, 89, 314
              ]
if __name__ == "__main__":
    np.random.seed(0)
    beta_parameters = [(2, 2), (4, 2), (2, 4), (4, 4), (8, 2), (2, 8), (8, 8)]
    for i, (alpha, beta) in enumerate(beta_parameters):
        print(f"Simulating for alpha={alpha}, beta={beta}")
        revenue = simulate(1000, 1000, alpha, beta)
        print(f"Average revenue: {revenue}")
        if revenue < THRESHOLDS[i]:
            raise Exception("The revenue is too low.")

    print("All tests passed.")