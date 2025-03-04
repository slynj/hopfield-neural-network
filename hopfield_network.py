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
        rho = np.sum([np.sum(p) for p in patterns]) / (num_data * self.num_neuron)

        # hebbian learning rule
        for i in tqdm(range(num_data)):
            p = patterns[i] - rho
            weights += np.outer(p, p)
        
        # w_{ii} = 0
        np.fill_diagonal(weights, 0)

        # normalization step => bc we keep adding the outer product
        weights /= num_data

        self.W = weights
    

    def energy(self, s):
        """ Calculates the energy of the given pattern based on the current weight matrix.

            The energy function is defined as:
                E = -0.5 * s^T (W * s)
            
            Lower energy => stable states.

        Args:
            s (numpy.ndarray): 1D numpy array represneting the state of the neurons.

        Returns:
            float: The calcualted energy of the given pattern.
        """
        # (-1/2) * s^T * W * s: 
        # same as: -0.5 * np.dot(np.dot(s, self.W), s) + np.sum(s * self.threshold)
        return -0.5 * s @ self.W @ s + np.sum(s * self.threshold)
    

    def predict(self, input_pattern, iteration=20, threshold=0, mode='sync'):
        """ Predicts the stable pattern for a given input.

        Args:
            input_pattern (numpy.ndarray): Given pattern to start prediction
            iteration (int, optional): Max num of iterations. Defaults to 20.
            tolerance (int, optional): Energy change threshold for convergence. Defaults to 0.
            mode (str, optional): Update mode. Defaults to 'sync'.

        Raises:
            ValueError: When mode is not 'sync' or 'async'

        Returns:
            numpy.ndarray: Predicted stable pattern after convergence (or the iteration).
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
        #  => update neuron state all at once
        if self.mode == 'sync':
            # init state and energy
            s = initial
            e = self.energy(s)
        
            # iteration
            for i in range(self.iteration):
                # update state and energy calc with new states
                s = np.sign((self.W @ s) - self.threshold) # np.sign => sign
                e_updated = self.energy(s)

                # state converged
                if e == e_updated:
                    return s
                
                e = e_updated

            return s

        # asynchronous update
        #  => one neuron at a time, randomly
        #  => slower
        elif self.mode == 'async':
            s = initial
            e = self.energy(s)

            for i in range(self.iteration):
                for j in range(100):
                    # select random neuron
                    index = np.random.randint(0, self.num_neuron)
                    # update state
                    s[index] = np.sign((self.W[index].T @ s) - self.threshold)

                # update state energy
                e_updated = self.energy(s)

                # state converged
                if e == e_updated:
                    return s
                
                e = e_updated
            
            return s

        # invalid mode
        else:
            raise ValueError("Invalid mode: Choose 'sync' or 'async'")