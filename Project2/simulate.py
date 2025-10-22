import numpy as np
from ID1_ID2_part2 import Recommender
import time

MAX_HORIZON = 15

def simulate_interaction(L, S, p):
    """_summary_

    Args:
        L (np.ndarray): A matrix in which the entry (i,j) represents the probability that a user of type j
                         will give a like to a clip from genre i.
        S (np.ndarray): A matrix in which the entry (i,j) represents the probability that a user of type j
                        won't leave the system after being recommended a clip from genre i and not liking it.
        p (np.ndarray): The prior over user types. The entry i represents the probability that a user is of type i.

    Returns:
        integer: The number of likes the user gave to the recommended clips until it left the system.
    """
    
    # sample user from the prior
    user = np.random.choice(range(len(p)), p=p)
    
    # initialize the recommender - time limit of 2 minutes
    initialization_start = time.time()
    recommender = Recommender(L, S, p)
    initialization_end = time.time()
    
    if initialization_end - initialization_start > 120:
        return 0
    
    #initialize the cumulative likes
    cumulative_likes = 0
    
    for t in range(MAX_HORIZON):
        # recommend an item - time is limited to 0.1 seconds
        recommendation_start = time.time()
        recommendation = recommender.recommend()
        recommendation_end = time.time()
        
        if recommendation_end - recommendation_start > 0.1:
            return 0
        
        # observe the user's response
        like = np.random.rand() < L[recommendation, user]

        # user of type j will stay in the system even though they don't like the item with probability S[i, j]
        stay = 1 if like else np.random.rand() < S[recommendation, user]
        
        if not stay:
            return cumulative_likes
        
        cumulative_likes += like            
        
        # update the recommender - time is limited to 0.1 seconds
        update_start = time.time()
        recommender.update(like)
        update_end = time.time()
        
        if update_end - update_start > 0.1:
            return 0
        
    return cumulative_likes

# Instance 1
L1 = np.array([[0.8, 0.7, 0.6], [0.79, 0.69, 0.59], [0.78, 0.68, 0.58]])
S1 = np.array([[0.56, 0.46, 0.36], [0.55, 0.45, 0.35], [0.54, 0.44, 0.34]])
p1 = np.array([0.35, 0.45, 0.2])

# Instance 2
L2 = np.array([[0.9, 0.75], [0.64, 0.5]])
S2 = np.array([[0.2, 0.4], [0.7, 0.8]])
p2 = np.array([0.3, 0.7])

# Instances 3a, 3b, 3c (same matrices L3, S3, different priors p3a, p3b, p3c)
L3 = np.array([[0.99, 0.2, 0.2], 
                [0.2, 0.99, 0.2], 
                [0.2, 0.2, 0.99], 
                [0.93, 0.93, 0.4],
                [0.4, 0.93, 0.93],
                [0.93, 0.4, 0.93],
                [0.85, 0.85, 0.85]])
S3 = np.zeros((7, 3))
p3a = np.array([0.9, 0.05, 0.05])
p3b = np.array([1/3, 1/3, 1/3])
p3c = np.array(object=[0.45, 0.25, 0.3])

# Instance 4
L4 = np.array([[0.94, 0.21, 0.02, 0.05, 0.86, 0.61, 0.59, 0.26],
               [0.91, 0.46, 0.87, 0.19, 0.64, 0.40, 0.83, 0.67],
               [0.25, 0.52, 0.32, 0.13, 0.15, 0.82, 0.46, 0.41],
               [0.10, 0.85, 0.70, 0.95, 0.06, 0.49, 0.68, 0.98]])
S4 = np.array([[0.51, 0.26, 0.98, 0.12, 0.99, 0.15, 0.74, 0.21],
               [0.92, 0.37, 0.17, 0.45, 0.81, 0.56, 0.28, 0.55],
               [0.61, 0.40, 0.21, 0.87, 0.25, 0.03, 0.85, 0.21],
               [0.62, 0.47, 0.06, 0.28, 0.90, 0.75, 0.48, 0.79]])
p4 = np.array([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8])

# Instance 5
L5 = np.array([[0.88, 0.12, 0.08, 0.29, 0.01, 0.34, 0.83, 0.61, 0.05, 0.07],
              [0.04, 0.01, 0.42, 0.24, 0.79, 0.24, 0.98, 0.88, 0.83, 0.38],
              [0.34, 0.76, 0.08, 0.07, 0.52, 0.43, 0.43, 0.82, 0.62, 0.88],
              [0.52, 0.58, 0.54, 0.59, 0.83, 0.79, 0.71, 0.72, 0.39, 0.28],
              [0.47, 0.49, 0.21, 0.51, 0.15, 0.22, 0.43, 0.56, 0.83, 0.04],
              [0.94, 0.73, 0.53, 0.54, 0.70, 0.79, 0.26, 0.21, 0.80, 0.56],
              [0.15, 0.72, 0.87, 0.83, 0.45, 0.90, 0.49, 0.45, 0.58, 0.95],
              [0.60, 0.23, 0.48, 0.74, 0.37, 0.90, 0.56, 0.82, 0.90, 0.86],
              [0.10, 0.57, 0.80, 0.47, 0.18, 0.91, 0.68, 0.52, 0.04, 0.42],
              [0.61, 0.11, 0.95, 0.39, 0.23, 0.13, 0.50, 0.10, 1.00, 0.26]])
S5 = np.array([[0.67, 0.83, 0.24, 0.07, 0.54, 0.15, 0.79, 0.44, 0.93, 0.49],
              [0.96, 0.23, 0.89, 0.54, 0.36, 0.43, 0.74, 0.32, 0.23, 0.88],
              [0.03, 0.88, 0.33, 0.79, 0.21, 0.10, 0.01, 0.62, 0.39, 0.86],
              [0.88, 0.84, 0.84, 0.65, 0.33, 0.44, 0.98, 0.85, 0.42, 0.42],
              [0.28, 0.45, 0.99, 0.25, 0.85, 0.16, 1.00, 0.87, 0.88, 0.82],
              [0.55, 0.81, 0.76, 0.25, 0.78, 0.80, 0.36, 0.37, 0.55, 0.75],
              [0.65, 0.94, 0.03, 0.32, 0.51, 0.89, 0.61, 0.89, 0.55, 0.96],
              [0.35, 0.03, 0.78, 0.96, 0.20, 0.44, 0.08, 0.82, 0.51, 0.28],
              [0.16, 0.57, 0.93, 0.81, 0.94, 0.48, 0.93, 0.35, 0.73, 0.37],
              [0.12, 0.42, 0.81, 0.25, 0.44, 0.99, 0.08, 0.51, 0.16, 0.38]])
p5 = np.array([0.11, 0.12, 0.07, 0.1, 0.05, 0.13, 0.1, 0.11, 0.11, 0.1])

if __name__ == "__main__":
    num_of_likes = simulate_interaction(L1, S1, p1)
    print(num_of_likes)




    