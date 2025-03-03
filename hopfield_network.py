import numpy as np
from tqdm import tqdm

class HopfieldNetwork:
    def __init__(self):
        self.num_neuron = 0
        self.W = np.array([0])
        self.iteration = 0
        self.threshold = 0
        self.mode = 'Null'
    

    def train(self, patterns):
        """ Training with Hebbian learning rule. 

            Updates the weight matrix based on the patterns. It computes the outer product of each 
            pattern with itself and accumulates the result to the weight matrix. Weight with itself
            is 0, and the matrix is normalized.

        Args:
            patterns (numpy.ndarray): An array of patterns for training. Pattern should be
            a numpy.ndarray with bipolar values corresponding to neuron states.
        """
        print("Training Patterns ... ")

        num_data = len(patterns)
        self.num_neuron = patterns[0].shape[0]

        # weight init
        weights = np.zeros((self.num_neuron, self.num_neuron))
        # avg neuron value used for stable training
        #  => subtracting rho from the data prevents values from leaning towards -1 or 1
        #  => if there's a pattern with all 1s, it's too strong so the result might always
        #     lead to that pattern even though there are other patterns. so it's more stable
        #     to subtract the avg neuron value = rho
        rho = np.sum([np.sum(p) for p in patterns] / (num_data * self.num_neuron)) 

        # hebbian learning rule
        for i in tqdm(range(num_data)):
            p = patterns[i] - rho
            weights += np.outer(p, p)
        
        # w_{ii} = 0
        np.fill_diagonal(weights, 0)

        # normalization step => bc we keep adding the outer product
        weights /= num_data

        self.W = weights
    

    # def sign(self, x, threshold=0.5):
    #     """ Activation Function, updates the neuron's state.

    #     Args:
    #         x (numpy.ndarray or float): Input value or array of values to be transformed.

    #     Returns:
    #         numpy.ndarray or int: Transformed input where each element is -1 or 1. Returns a single int if input is a scaler.
    #     """
    #     # ternery => convert nums to bipolar
    #     return np.where(x > threshold, 1, -1)
    

    def energy(self, s):
        """ Calculates the energy of the given pattern based on the current weight matrix.

            The energy function is defined as:
                E = -0.5 * s^T (W * s)
            
            Lower energy => stable states.

        Args:
            pattern (numpy.ndarray): 1D numpy array represneting the state of the neurons.

        Returns:
            float: The calcualted energy of the given pattern.
        """
        # (-1/2) * s^T * W * s: 
        # same as: -0.5 * np.dot(np.dot(s, self.W), s) + np.sum(s * self.threshold)
        return -0.5 * s @ self.W @ s + np.sum(s * self.threshold)


    

    def predict(self, input_pattern, iteration=20, threshold=0, mode='sync'):
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
        print("Predicting Pattern ... ")
        
        self.iteration = iteration
        self.threshold = threshold
        self.mode = mode

        # copy to avoid call by ref
        cp_input = np.copy(input_pattern)

        # predicted list
        predicted = []

        for i in tqdm(range(len(cp_input))):
            predicted.append(self._run(cp_input[i]))
        
        return predicted
    
    
    def _run(self, initial):
        # synchronous update
        if self.mode == 'sync':
            # init state and energy
            s = initial
            e = self.energy(s)
        
        # iteration
        for i in range(self.iteration):
            # update state and energy calc with new states
            s = np.sign((self.W @ s) - self.threshold) # np.sign => sign
            

        pattern = np.array(input_pattern)
        prev_energy = self.energy(pattern)

        # until it converges (unless it hits 100 times)
        for step in range(max_steps):
            # calculate everything, then update at once
            if mode == 'sync':
                updated_pattern = pattern.copy()

                for i in range(self.size): # for each neurons
                    raw_value = np.dot(self.W[i], pattern)
                    updated_pattern[i] = self.sign(raw_value)

                pattern = updated_pattern

            # update each neurons one by one
            elif mode == 'async':
                for i in range(self.size):
                    raw_value = np.dot(self.W[i], pattern)
                    pattern[i] = self.sign(raw_value)
            
            else:
                raise ValueError("Invalid mode: Choose 'sync' or 'async'")
            
            # calculate the energy again after the update
            curr_energy = self.energy(pattern)

            if abs(curr_energy - prev_energy) <= tolerance:
                print(f"Converged after {step + 1} steps with energy {curr_energy}.")
                break
        
            prev_energy = curr_energy


        return pattern