import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        """ Init weight matrix.

        Args:
            size (int): # of neurons in the network. Determines the dimension of the weight matrix.
        """
        self.size = size
        self.weights = np.zeros((size, size))
    

    def train(self, patterns):
        """ Training with Hebbian learning rule. 

            Updates the weight matrix based on the patterns. It computes the outer product of each 
            pattern with itself and accumulates the result to the weight matrix. Weight with itself
            is 0, and the matrix is normalized.

        Args:
            patterns (array-like): A list or array of patterns for training. Pattern should be
            a array-like struct (list, numpy array, etc) with bipolar values corresponding to
            neuron states.
        """
        for p in patterns:
            p = np.array(p)
            # computes the outer product of vector p and p
            #  => outer product of 2 vectors = a * b transposed
            #  => square matrix, each elem is the product of 2 elements from p
            #     outerP[i][j] = p[i] * p[j]
            self.weights += np.outer(p, p)
        
        np.fill_diagonal(self.weights, 0) # w_{ii} = 0

        # normalization step => bc we keep adding the outer product
        self.weights /= (len(patterns) * self.size)
    

    def sign(self, x):
        """ Activation Function, updates the neuron's state.

        Args:
            x (numpy.ndarray or float): Input value or array of values to be transformed.

        Returns:
            numpy.ndarray or int: Transformed input where each element is -1 or 1. Returns a single int if input is a scaler.
        """
        # ternery => convert nums to bipolar
        return np.where(x >= 0, 1, -1)
    

    def energy(self, pattern):
        """ Calculates the energy of the given pattern based on the current weight matrix.

            The energy function is defined as:
                E = -0.5 * s^T (W * s)
            
            Lower energy => stable states.

        Args:
            pattern (numpy.ndarray): 1D numpy array represneting the state of the neurons.

        Returns:
            float: The calcualted energy of the given pattern.
        """
        # pattern.T is pattern tansposed
        return -0.5 * np.dot(pattern.T, np.dot(self.weights, pattern))
    

    def predict(self, input_pattern, max_steps=100, tolerance=0, mode='sync'):
        """ Predicts the stable pattern for a given input.

        Args:
            input_pattern (array-like): Given pattern to start prediction
            max_steps (int, optional): Max num of iterations. Defaults to 100.
            tolerance (int, optional): Energy change threshold for convergence. Defaults to 0.
            mode (str, optional): Update mode. Defaults to 'sync'.

        Raises:
            ValueError: When mode is not 'sync' or 'async'

        Returns:
            numpy.ndarray: Predicted stable pattern after convergence (or the max steps).
        """
        pattern = np.array(input_pattern)
        prev_energy = self.energy(pattern)

        # until it converges (unless it hits 100 times)
        for step in range(max_steps):
            # calculate everything, then update at once
            if mode == 'sync':
                updated_pattern = pattern.copy()

                for i in range(self.size): # for each neurons
                    raw_value = np.dot(self.weights[i], pattern)
                    pattern[i] = self.sign(raw_value)

                pattern = updated_pattern

            # update each neurons one by one
            elif mode == 'async':
                for i in range(self.size):
                    raw_value = np.dot(self.weights[i], pattern)
                    pattern[i] = self.sign(raw_value)
            
            else:
                raise ValueError("Invalid mode: Choose 'sync' or 'async'")
            
            # calculate the energy again after the update
            curr_energy = self.energy(pattern)

            if abs(curr_energy - prev_energy) <= tolerance:
                print(f"Converged after {step + 1} steps with energy {curr_energy}.")
                print(curr_energy)
                print(prev_energy)
                break
        
            prev_energy = curr_energy

        return pattern