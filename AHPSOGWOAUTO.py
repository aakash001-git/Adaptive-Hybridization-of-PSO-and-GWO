import numpy as np


np.random.seed(42)


class NEWAHPSOGWOAUTO:
    def __init__(self, fitness_func, dimension=30, pop_size=20, max_iter=100, c1=0.5, c2=0.5, w=0.9):
        self.fitness_func = fitness_func
        self.dimension = dimension
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.c1 = c1  # PSO cognitive coefficient
        self.c2 = c2  # PSO social coefficient
        self.w = w  # Inertia weight
       
        self.X = np.random.rand(pop_size, dimension)  # Particles' positions
        self.V = np.random.rand(pop_size, dimension)  # Particles' velocities
        self.p_best = np.copy(self.X)  # Personal best positions
        self.g_best = np.copy(self.X[0])  # Placeholder for global best
        self.fitness_p_best = np.full(pop_size, np.inf)
        self.fitness_g_best = np.inf


        # GWO components
        self.alpha = np.zeros(dimension)
        self.beta = np.zeros(dimension)
        self.delta = np.zeros(dimension)
        self.alpha_score = np.inf
        self.beta_score = np.inf
        self.delta_score = np.inf
       
        self.evolution = []  # For tracking the fitness progression


    def update_gwo_leaders(self):
        for i in range(self.pop_size):
            fitness = self.fitness_func(self.X[i, :])
           
            if fitness < self.alpha_score:
                self.delta_score, self.delta = self.beta_score, np.copy(self.beta)
                self.beta_score, self.beta = self.alpha_score, np.copy(self.alpha)
                self.alpha_score, self.alpha = fitness, np.copy(self.X[i, :])
            elif fitness < self.beta_score:
                self.delta_score, self.delta = self.beta_score, np.copy(self.beta)
                self.beta_score, self.beta = fitness, np.copy(self.X[i, :])
            elif fitness < self.delta_score:
                self.delta_score, self.delta = fitness, np.copy(self.X[i, :])


    def calculate_X_alpha_beta_delta(self):
        # Dynamic weights for alpha, beta, delta based on fitness
        omega_alpha = 1 / (1 + self.alpha_score)
        omega_beta = 1 / (1 + self.beta_score)
        omega_delta = 1 / (1 + self.delta_score)
        total_omega = omega_alpha + omega_beta + omega_delta
       
        X_alpha_beta_delta = (
            (omega_alpha * self.alpha + omega_beta * self.beta + omega_delta * self.delta) / total_omega
        )
        return X_alpha_beta_delta


    def update_p_best(self):
        for i in range(self.pop_size):
            fitness = self.fitness_func(self.X[i, :])
            if fitness < self.fitness_p_best[i]:
                self.fitness_p_best[i] = fitness
                self.p_best[i] = np.copy(self.X[i, :])


    def update_velocity(self, X_alpha_beta_delta, lambda_param):
        for i in range(self.pop_size):
            r1, r2 = np.random.random(), np.random.random()
            self.V[i] = (
                self.w * self.V[i]
                + lambda_param * (
                    self.c1 * r1 * (self.p_best[i] - self.X[i])
                    + self.c2 * r2 * (X_alpha_beta_delta - self.X[i])
                )
            )


    def update_positions(self, lambda_param):
        for i in range(self.pop_size):
            # GWO encircling effect
            delta_X_GWO = (self.alpha + self.beta + self.delta) / 3 - self.X[i]
            self.X[i] += self.V[i] + (1 - lambda_param) * delta_X_GWO


    def opt(self):
        for t in range(self.max_iter):
            # Update GWO leaders
            self.update_gwo_leaders()
           
            # Calculate combined influence of GWO leaders
            X_alpha_beta_delta = self.calculate_X_alpha_beta_delta()
           
            # Update personal and global bests
            self.update_p_best()
            best_particle_idx = np.argmin(self.fitness_p_best)
            if self.fitness_p_best[best_particle_idx] < self.fitness_g_best:
                self.fitness_g_best = self.fitness_p_best[best_particle_idx]
                self.g_best = np.copy(self.p_best[best_particle_idx])


            # Adaptive λ: Decreases from 1 to 0.5
            lambda_param = 1 - (t / self.max_iter) * 0.5
           
            # Update velocity and positions
            self.update_velocity(X_alpha_beta_delta, lambda_param)
            self.update_positions(lambda_param)
           
            # Track evolution
            self.evolution.append(self.fitness_g_best)


    def return_result(self):
        return np.array(self.evolution)