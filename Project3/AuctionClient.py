import copy
from time import perf_counter
from scipy.stats import beta
import numpy as np


class AuctionClient:
    def __init__(self, value, clients_num, insurances_num):
        # Main attributes
        self.client_value = value  # Value of the insurance for this client
        self.total_clients = clients_num  # Total number of clients in the auction
        self.total_insurances = insurances_num  # Total number of insurances available

        # List of past winning bids for previous rounds
        self.past_winning_bids = []

        # Initial estimated values of the competitor clients
        self.estimated_competitor_values = beta.rvs(5, 2, size=self.total_clients - 1)

        # Predicted durations for each insurance auction round
        self.predicted_round_durations = 3 * beta.rvs(2, 5, size=self.total_insurances)

        # Default utility if no bid is made by the client
        self.default_utility = 0.5 * self.client_value

        # Case 1: More insurances available than competitors
        self.more_insurances_than_clients = False

        # Case 2: More competitors than available insurances
        self.more_clients_than_insurances = False

        # Determine if we are in Case 1 or Case 2
        if insurances_num > clients_num:
            self.more_insurances_than_clients = True
        else:
            self.more_clients_than_insurances = True



    def decide_bid(self, t, duration):

        # In the first round, no bid is made
        if t == 0:
            return -1

        # Find the closest predicted duration to the actual duration of the round and remove it
        closest_predicted_duration = min(self.predicted_round_durations,
                                         key=lambda x: abs(x - duration))
        self.predicted_round_durations = np.delete(self.predicted_round_durations,
                                                   np.where(
                                                       self.predicted_round_durations == closest_predicted_duration))

        future_profits = []
        for i, predicted_duration in enumerate(self.predicted_round_durations):
            future_profit = self.client_value * predicted_duration
            future_profits.append(future_profit)

        # Mean future profit among predicted rounds
        if len(future_profits) != 0:
            mean_future_profit = sum(future_profits) / len(future_profits)
        else:
            mean_future_profit = 0

        current_profit = self.client_value * duration

        remaining_competitors = self.total_clients - len(self.past_winning_bids) - 1


        # Case 1: More insurances than competitors
        if self.more_insurances_than_clients==True:

            if t == self.total_insurances - 1 :
                if duration < 0.5:
                    return -1
                else :
                    return self.client_value

            # If multiple competitors remain, wait and do not bid
            if remaining_competitors > 1:
                return -1
            elif remaining_competitors == 1:
                # Make a decision to bid or wait based on future vs current profit comparison
                if ((current_profit >= mean_future_profit and duration > 1.65) or duration > 1.8) :
                    return self.client_value
                else:
                    return -1
            # If no competitors remain, compare current and future profits
            else:
                max_future_profit = self.client_value*max(d for d in self.predicted_round_durations)
                remaining_insurances = self.total_insurances - t - 1
                if ((current_profit >= max_future_profit and remaining_insurances > 2) or duration > 1.6):
                    return self.client_value
                elif (current_profit >= max_future_profit*0.85 and remaining_insurances <= 2):
                    return self.client_value
                else:
                    return -1

        # Case 2: More competitors than available insurances or equal
        else :
            # Retrieve the estimated values of the remaining competitors
            estimated_values_sorted = sorted(self.estimated_competitor_values, reverse=True)

            if t > 0:
                next_competitor_value = self.past_winning_bids[-1]
            else:
                next_competitor_value = estimated_values_sorted[0]

            # Check if this is the last round
            is_final_round = (t + 1) == self.total_insurances
            # Decision logic based on competitor's value and current profit

            if self.client_value - next_competitor_value > 0.05:
                current_profit = (self.client_value - next_competitor_value) * duration

                # In the final round, ensure profit exceeds default utility
                if is_final_round:
                    if current_profit > self.default_utility:
                        return self.client_value
                    else:
                        return 0
                # Not in the final round
                else:
                    # Decide whether to bid or wait based on profit comparison
                    if current_profit >= mean_future_profit and current_profit >= self.default_utility:
                        return self.client_value
                    else:
                        return 0
            else:
                return 0

    def update(self, t, price):
        # Add the current round's winning bid to the list of past winning values
        self.past_winning_bids.append(price)


        # Calculate the number of remaining competitors
        remaining_competitors = self.total_clients - len(self.past_winning_bids) - 1

        # Only adjust estimated values if there are more than one remaining competitors
        if remaining_competitors > 1:
            # Find the closest estimated value to the winning bid price and remove it
            closest_estimated_value = min(self.estimated_competitor_values, key=lambda x: abs(x - price))
            self.estimated_competitor_values = np.delete(self.estimated_competitor_values,
                                                         np.where(
                                                             self.estimated_competitor_values == closest_estimated_value))
            # Sort the remaining estimated values
            self.estimated_competitor_values = np.sort(self.estimated_competitor_values)

            # Find the maximum value of the remaining estimates
            max_estimate = np.max(self.estimated_competitor_values)

            # Adjust the estimated values by scaling them based on the winning bid price and max estimate
            adjustment_factor = (price / max_estimate) * 0.9 if max_estimate > 0 else 1
            self.estimated_competitor_values = self.estimated_competitor_values * adjustment_factor




def auction_client_creator(value1, clients_num1, insurances_num1):
    return AuctionClient(value1, clients_num1, insurances_num1)


class NaiveAuctionClient:
    def __init__(self, value, clients_num, insurances_num):
        self.value = value
        self.clients_num = clients_num
        self.manufacturers_num = insurances_num

    def decide_bid(self, t, quality):
        return self.value

    def update(self, t, price):
        pass


def naive_auction_client_creator(value, clients_num, insurances_num):
    return NaiveAuctionClient(value, clients_num, insurances_num)


def simulate_single_auction(num_of_competitors, number_of_insurances):
    start = perf_counter()
    your_value = np.random.beta(5, 2)
    your_client = auction_client_creator(your_value, num_of_competitors + 1, number_of_insurances)
    end = perf_counter()
    if end - start > 2:
        raise Exception("The initialization of the client took too long.")

    competing_clients_list = [
        naive_auction_client_creator(np.random.beta(5, 2), num_of_competitors, number_of_insurances) for _ in
        range(num_of_competitors)]

    active_competing_clients = copy.deepcopy(competing_clients_list)
    for t in range(number_of_insurances):
        duration = 3 * np.random.beta(2, 5)

        start = perf_counter()
        your_bid = your_client.decide_bid(t, duration)
        end = perf_counter()
        if end - start > 0.5:
            raise Exception("The decision of the bid took too long.")

        competing_bid_list = [client.decide_bid(t, duration) for client in active_competing_clients]

        if len(active_competing_clients) == 0:
            if your_bid == -1:  # you didn't want to bid
                continue
            else:
                return duration * your_value

        if your_bid == -1:  # you didn't want to bid
            if len(active_competing_clients) == 1:
                second_highest_bid = 0
                active_competing_clients.pop(0)
            else:
                highest_bid = max(competing_bid_list)
                clients_with_highest_bid = [i for i, bid in enumerate(competing_bid_list) if bid == highest_bid]
                winner = np.random.choice(clients_with_highest_bid)
                second_highest_bid = max([bid for i, bid in enumerate(competing_bid_list) if i != winner])
                active_competing_clients.pop(winner)
        else:
            highest_bid = max(your_bid, max(competing_bid_list))
            clients_with_highest_bid = [-1] if your_bid == highest_bid else []
            clients_with_highest_bid += [i for i, bid in enumerate(competing_bid_list) if bid == highest_bid]
            winner = np.random.choice(clients_with_highest_bid)
            second_highest_bid = your_bid if winner != -1 else 0
            less_then_maximum_scores = [bid for i, bid in enumerate(competing_bid_list) if i != winner]
            if len(less_then_maximum_scores) > 0:
                second_highest_bid = max(second_highest_bid, max(less_then_maximum_scores))

            if winner == -1:
                return duration * (your_value - second_highest_bid)
            else:
                active_competing_clients.pop(winner)

        start = perf_counter()
        your_client.update(t, second_highest_bid)
        end = perf_counter()
        if end - start > 0.5:
            raise Exception("The update of the client took too long.")
        for client in active_competing_clients:
            client.update(t, second_highest_bid)

    return 0.5 * your_value  # if the simulation ends and you didn't win, you get default utility.


def simulate(simulations, num_of_competitors, number_of_insurances):
    revenues_list = []
    for _ in range(simulations):
        revenue = simulate_single_auction(num_of_competitors, number_of_insurances)
        revenues_list.append(revenue)

    return np.mean(revenues_list)


VARIABLES_VALUES = [(10, 5), (5, 10), (10, 10), (20, 10), (10, 20), (20, 20)]
BASELINES = [0.3565, 0.756, 0.3565, 0.3565, 0.856, 0.3565]

if __name__ == "__main__":
    np.random.seed(0)
    for i, (num_of_competitors, number_of_insurances) in enumerate(VARIABLES_VALUES):
        result = simulate(1000, num_of_competitors, number_of_insurances)
        if result < BASELINES[i]:
            raise Exception("You didn't beat the baseline.")

    print("All simulations passed.")