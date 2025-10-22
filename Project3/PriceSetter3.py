import numpy as np
from time import perf_counter
from scipy.stats import beta

ALPHA_VALUES = [i for i in range(1, 11)]
BETA_VALUES = [i for i in range(1, 11)]


class PriceSetter3:
    def __init__(self, rounds):
        """
        Initialize the price setter.
        In this settings, the values of the costumers is distributed according to a beta distribution with unknown parameters.

        Args:
            rounds (int): the number of rounds to simulate
        """
        self.rounds = rounds
        # Initialize alpha and beta parameters for Thompson Sampling
        self.alpha = 1
        self.beta = 1

    def set_price(self, t):
        """
        Return the price at time t.

        Args:
            t (int): the time period

        Returns:
            float: the price at time t
        """
        price = np.random.beta(self.alpha, self.beta)
        return price*0.8

    def update(self, t, outcome):
        """
        Update the price setter based on the outcome of the previous period.

        Args:
            t (int): the time period
            outcome (int): the outcome of the previous period - true if the product was sold, false otherwise
        """
        if t<self.rounds*0.7:
            # Update alpha and beta parameters based on the outcome
            if outcome == 1:  # If the product was sold, update alpha
                self.alpha += 0.9
            else:  # If the product was not sold, update beta
                self.beta += 1


def simulate(simulations, rounds):
    """
    Simulate the game for the given number of rounds.

    Args:
        rounds (int): the number of rounds to simulate the game

    Returns:
        float: the average revenue of the seller
    """

    simulations_results = []
    for _ in range(simulations):

        alpha = np.random.choice(ALPHA_VALUES)
        beta = np.random.choice(BETA_VALUES)
        start = perf_counter()
        price_setter = PriceSetter3(rounds)
        end = perf_counter()
        if end - start > 1:
            raise Exception("The initialization of the price setter is too slow.")
        revenue = 0

        for t in range(rounds):
            costumer_value = np.random.beta(alpha, beta)
            start = perf_counter()
            price = price_setter.set_price(t)
            end = perf_counter()
            if end - start > 0.3:
                raise Exception("The set_price method is too slow.")
            if costumer_value >= price:
                revenue += price

            start = perf_counter()
            price_setter.update(t, costumer_value >= price)
            end = perf_counter()
            if end - start > 0.3:
                raise Exception("The update method is too slow.")

        simulations_results.append(revenue)

    return np.mean(simulations_results)


if __name__ == "__main__":
    np.random.seed(0)
    print(simulate(1000, 1000))
