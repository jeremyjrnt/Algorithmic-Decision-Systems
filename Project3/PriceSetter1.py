import numpy as np
from time import perf_counter


class PriceSetter1:

    def __init__(self, rounds):
        """
        Initialize the price setter.
        In this settings, the values of the costumers is constant and unknown in advance.

        Args:
            rounds (int): the number of rounds to simulate
        """

        self.interval_lower = float(0)
        self.interval_upper = float(1)
        self.h = (self.interval_upper - self.interval_lower)/float(2)
        self.percent = 0.01
        self.last_v = None
        self.k = int(rounds*self.percent)


    def set_price(self, t):
        """
        Return the price at time t.

        Args:
            t (int): the time period

        Returns:
            float: the price at time t
        """
        #print("tour : ", t)
        #print("[ " + str(self.interval_lower) + ", " + str(self.interval_upper) + " ]")
        #print("Le prix proposé est : ", self.h)
        #print()
        return self.h

    def update(self, t, outcome):
        """
        Update the price setter based on the outcome of the previous period.

        Args:
            t (int): the time period
            outcome (int): the outcome of the previous period - true if the product was sold, false otherwise
        """
        #print("Le client a acheté au tour précedent: ", outcome)

        if outcome==True and not None :
            self.last_v = self.h

        if t < self.k  :
            if outcome == True:
                self.interval_lower = self.h
            else:
                self.interval_upper = self.h
            self.h = self.interval_lower + ((self.interval_upper - self.interval_lower) / float(2))
        elif self.last_v is not None:
            self.h = self.last_v
        else:
            self.h= 0.5




def simulate(simulations, rounds):
    """
    Simulate the game for the given number of rounds.

    Args:
        rounds (int): the number of rounds to simulate

    Returns:
        float: the revenue of the price setter
    """
    simulations_results = []
    for _ in range(simulations):
        start = perf_counter()
        price_setter = PriceSetter1(rounds)
        end = perf_counter()
        if end - start > 1:
            raise Exception("The initialization of the price setter is too slow.")
        revenue = 0
        costumer_value = np.random.uniform(0, 1)

        for t in range(rounds):
            start = perf_counter()
            price = price_setter.set_price(t)
            end = perf_counter()
            if end - start > 0.1:
                raise Exception("The set_price method is too slow.")
            if costumer_value >= price:
                revenue += price

            start = perf_counter()
            price_setter.update(t, costumer_value >= price)
            end = perf_counter()
            if end - start > 0.1:
                raise Exception("The update method is too slow.")

        simulations_results.append(revenue)

    return np.mean(simulations_results)


if __name__ == "__main__":
    np.random.seed(0)
    print(simulate(1000, 1000))