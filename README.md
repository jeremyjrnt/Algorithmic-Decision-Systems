# Algorithmic Decision Systems

This repository contains three projects in the field of algorithmic decision systems, each addressing different problems in game theory, optimization, and machine learning.

## Project 1: Social Network Influence Marketing Optimization (NoseBook)

### Objective

Develop an algorithm to select the most effective influencers in a social network to maximize product exposure under budget constraints.

### Problem Statement

In a simulated social network called "NoseBook", we need to choose a set of initial influencers with a maximum budget of 1000 units to maximize the product exposure score after 6 rounds of propagation.

### Methods Used

#### 1. Network Analysis with NetworkX

- **Directed Graph Construction**: Creation of a DAG (Directed Acyclic Graph) from friendship relationships
- **Node Filtering**: Removal of nodes with less than 5 outgoing connections AND less than 3 ancestors
- **Inverse Graph**: Construction of G_inv to calculate distances in reverse direction

#### 2. Centrality Measures

- **Betweenness Centrality**: Measures the importance of a node as an intermediary in shortest paths
- **Eigenvector Centrality**: Evaluates influence based on connection quality
- **Adjusted Ancestor Count**: Number of ancestors accessible within a 6-step radius

#### 3. Multi-Objective Optimization

- **Normalization and Cost Adjustment**: Each centrality measure is normalized then divided by the influencer's cost
- **Weighted Combination**:
  - Ancestors: 25%
  - Betweenness: 25%
  - Eigenvector: 50%

#### 4. Stochastic Optimization Algorithm

- **Random Sampling**: 5000 trials to find the best influencer combination
- **Budget Constraint**: Selection of influencers respecting the 1000-unit budget
- **Monte Carlo Simulation**: 100 simulations per configuration to estimate expectation

### Underlying Theory

- **Graph Theory**: Analysis of social network structure
- **Information Diffusion**: Modeling viral propagation
- **Combinatorial Optimization**: Optimal selection under constraints
- **Centrality Analysis**: Identification of most influential nodes

### Process Simulation

The model simulates propagation over 6 rounds where each user purchases with probability proportional to the number of friends who have already purchased.

---

## Project 2: Adaptive Recommendation System

### Part 1: Viewing Duration Prediction (Collaborative Filtering)

#### Objective

Predict user viewing duration for video clips using two collaborative filtering approaches.

#### Implemented Methods

##### 1.1 Bias Model

**Formula**: $\hat{r}_{ui} = \mu + b_u + b_i$

Where:

- $\mu$: global average of viewing durations
- $b_u$: user bias (personal tendency)
- $b_i$: item bias (clip popularity)

**Optimization**:

- **Cost Function**: $\min \sum_{(u,i)} (r_{ui} - \hat{r}_{ui})^2 + \lambda(||b_u||^2 + ||b_i||^2)$
- **L2 Regularization**: $\lambda = 0.1$
- **Gradient Descent**: Learning rate = 1.75e-3, 1000 iterations

##### 1.2 Singular Value Decomposition (SVD)

**Principle**: Low-rank approximation of the user-item matrix

**Process**:

1. Construction of sparse matrix $R_{users \times clips}$
2. SVD decomposition: $R = U\Sigma V^T$
3. Approximation with $k=20$ principal components
4. Reconstruction: $\hat{R} = U_k\Sigma_k V_k^T$

### Part 2: Multi-Armed Bandit Recommendation System

#### Objective

Develop an adaptive recommender that learns unknown user preferences over 15 rounds to maximize "likes".

#### Theoretical Framework: Contextual Multi-Armed Bandits

- **States**: Unknown user types
- **Actions**: Clip genres to recommend
- **Rewards**: Received likes and user retention

#### Implemented Algorithm

##### Bayesian Inference

- **Belief Update**: Using Bayes' rule to estimate user type
- **Weighted Likelihood**: Application of weight factor (1.35) to accelerate learning

##### Adaptive Strategy

1. **Exploration Phase** (rounds 1-14): Selection based on future reward expectation
2. **Final Exploitation** (round 15): Choose best genre for most probable user type

##### Performance Simulation

Pre-computation of expected rewards over 15 rounds via Monte Carlo simulation for each (genre, user_type) pair.

### Underlying Theory

- **Collaborative Filtering**: Exploitation of user-item similarities
- **Matrix Decomposition**: Dimensionality reduction
- **Multi-Armed Bandits**: Exploration-exploitation balance
- **Bayesian Inference**: Belief updating

---

## Project 3: Algorithmic Pricing Systems

### General Objective

Develop dynamic pricing algorithms for different market scenarios, optimizing revenue in incomplete information environments.

### PriceSetter1: Unknown and Constant Customer Value

#### Problem Statement

A seller facing a single customer whose product value is unknown but constant over time.

#### Method: Adaptive Binary Search

1. **Exploration Phase** (1% of rounds):
   - Binary search to bracket customer value
   - Update interval $[v_{min}, v_{max}]$ based on purchase responses
2. **Exploitation Phase**:
   - Use last accepted price as optimal price

#### Theory

- **Dichotomous Search**: Guaranteed convergence to true value
- **Exploration-exploitation trade-off**: Regret minimization

### PriceSetter2: Known Beta Distribution

#### Problem Statement

Customer values follow a known Beta($\alpha, \beta$) distribution. Find optimal fixed price.

#### Method: Analytical Optimization

**Objective Function**: $\max_p p \cdot P(V \geq p)$

Where $P(V \geq p) = 1 - F_{Beta}(p; \alpha, \beta)$

**Implementation**:

- Price space discretization (step size 0.001)
- Expected revenue calculation for each price
- Selection of price maximizing $p \times (1 - CDF_{Beta}(p))$

#### Theory

- **Revenue Optimization**: Balance between high price and sale probability
- **Probability Distributions**: Exploitation of Beta distribution properties

### PriceSetter3: Unknown Beta Distribution

#### Problem Statement

Beta distribution parameters are unknown and must be learned online.

#### Method: Adaptive Thompson Sampling

1. **Bayesian Modeling**:

   - Prior parameters $\alpha, \beta$ updated based on observations
   - Binomial likelihood (purchase/no purchase)
2. **Pricing Strategy**:

   - Sampling from posterior distribution
   - Price = 80% of sampled value (conservative strategy)
3. **Update** (first 70% of rounds):

   - Purchase → $\alpha += 0.9$
   - No purchase → $\beta += 1$

#### Theory

- **Thompson Sampling**: Optimal sampling for bandits
- **Bayesian Inference**: Belief updating
- **Bayesian Regret**: Expected regret minimization

### Common Framework: Game Theory and Mechanisms

#### Transversal Concepts

- **Asymmetric Information**: Seller doesn't know buyer preferences
- **Online Learning**: Real-time adaptation based on observations
- **Optimization under Uncertainty**: Decision-making in stochastic environments
- **Selling Mechanisms**: Optimal pricing strategy design

#### Performance Metrics

- **Regret**: Difference between optimal and realized revenue
- **Convergence Rate**: Speed of parameter learning
- **Robustness**: Performance across different customer distributions

---

## Technologies and Tools Used

### Python Libraries

- **NetworkX**: Graph analysis and social networks
- **NumPy/SciPy**: Numerical computing and optimization
- **Pandas**: Data manipulation
- **Matplotlib**: Data visualization
- **scikit-learn**: Machine learning algorithms

### Implemented Algorithms

- **Stochastic Optimization**: Random search, gradient descent
- **Matrix Decomposition**: SVD, low-rank approximation
- **Multi-Armed Bandits**: Thompson Sampling, UCB
- **Game Theory**: Equilibria, dominant strategies

### Mathematical Concepts

- **Linear Algebra**: SVD, matrices, projections
- **Probability**: Distributions, Bayesian inference
- **Optimization**: Convex programming, local search
- **Graph Theory**: Centrality, paths, connectivity

## Conclusion

These three projects illustrate the practical application of fundamental theoretical concepts in decision-making computer science:

1. **Combinatorial optimization** for influencer selection
2. **Machine learning** for recommendation systems
3. **Game theory** for algorithmic pricing
