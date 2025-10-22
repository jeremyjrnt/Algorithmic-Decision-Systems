from datetime import time
import numpy as np


# HELP STATIC FUNCTIONS

def simulations(L, S, num_tours, num_simulations=100):
    num_genres, num_user_types = L.shape
    expected_likes = np.zeros((num_genres, num_user_types))

    for genre in range(num_genres):
        for user_type in range(num_user_types):
            total_likes = 0

            for _ in range(num_simulations):
                likes = 0
                for _ in range(num_tours):
                    if np.random.rand() < L[genre, user_type]:
                        likes += 1
                    else:
                        if np.random.rand() >= S[genre, user_type]:
                            break

                total_likes += likes

            expected_likes[genre, user_type] = total_likes / num_simulations

    return expected_likes


def find_strategic_dominance(matrix, threshold=1.05):
    num_genres, num_users = matrix.shape

    for i in range(num_genres):
        dominated = False
        for j in range(num_genres):
            if i != j:
                # Check if genre i dominates genre j with the threshold
                if all(matrix[i, :] >= threshold * matrix[j, :]):
                    continue
                else:
                    dominated = True
                    break
        if not dominated:
            return i  # Found a genre with strong strategic dominance
    return None  # No genre with strong strategic dominance


def find_nash_equilibrium(matrix, columns):
    # Extract the submatrix with the specified columns
    submatrix = matrix[:, columns]

    # Calculate the sum of each row in the submatrix
    row_sums = submatrix.sum(axis=1)

    # Find the index of the row with the maximum sum
    nash_equilibrium_row = np.argmax(row_sums)

    return nash_equilibrium_row

'''The Recommender Class'''
class Recommender:
    # Your recommender system class

    def __init__(self, L, S, p):
        """_summary_

        Args:
        L (np.ndarray): A matrix in which the entry (i,j) represents the probability that a user of type j
                         will give a like to a clip from genre i.
        S (np.ndarray): A matrix in which the entry (i,j) represents the probability that a user of type j
                        won't leave the system after being recommended a clip from genre i and not liking it.
        p (np.ndarray): The prior over user types. The entry i represents the probability that a user is of type i."""

        self.L = L
        self.S = S
        self.p = p

        self.current_round = 0
        self.genre_recommendation = None
        self.current_posterior = p.copy()
        self.weight_factor = 1.35

        self.B = simulations(self.L, self.S, 15)


    def recommend(self):
        """_summary_

        Returns:
        integer: The index of the clip that the recommender recommends to the user."""

        if self.current_round == 14:
            believe = np.argmax(self.current_posterior)
            return np.argmax(self.L[:, believe])
        else:
            self.genre_recommendation = np.argmax(np.dot(self.B, self.current_posterior))
            return self.genre_recommendation



    def update(self, signal):
        """_summary_

        Args:
        signal (integer): A binary variable that represents whether the user liked the recommended clip or not.
                          It is 1 if the user liked the clip, and 0 otherwise."""

        self.current_round += 1


        # Calculate the new posterior probabilities
        for j in range(len(self.current_posterior)):

            if signal == 1:
                likelihood = self.L[self.genre_recommendation, j]
            else:
                likelihood = (1 - self.L[self.genre_recommendation, j])

            # Apply the weight factor to the likelihood
            weighted_likelihood = likelihood ** self.weight_factor

            self.current_posterior[j] *= weighted_likelihood

            # Normalize the posterior to make sure it sums to 1
            self.current_posterior /= np.sum(self.current_posterior)



# an example of a recommender that always recommends the item with the highest probability of being liked
class GreedyRecommender:
    def __init__(self, L, S, p):
        self.L = L
        self.S = S
        self.p = p

    def recommend(self):
        return np.argmax(np.dot(self.L, self.p))

    def update(self, signal):
        pass


