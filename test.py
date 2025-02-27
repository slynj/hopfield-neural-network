import numpy as np
from hopfield_network import HopfieldNetwork

network_size = 5
hnet = HopfieldNetwork(network_size)

patterns = [
    [1, -1, 1, -1, 1],
    [-1, 1, -1, 1, -1]
]

hnet.train(patterns)
print("Training Completed.\n")


def pattern_gen(n=network_size):
    return np.random.choice([-1, 1], size=n)


def sync_testing(test_pattern):
    print(f"Input Pattern: {test_pattern}\n")
    
    predicted_sync = hnet.predict(test_pattern, mode='sync')
    print(f"Predicted Pattern (sync): {predicted_sync}\n\n")


def async_testing(test_pattern):
    print(f"Input Pattern: {test_pattern}\n")
    
    predicted_async = hnet.predict(test_pattern, mode='async')
    print(f"Predicted Pattern (async): {predicted_async}\n\n")


def testing(fn, n):
    [fn(pattern_gen()) for _ in range(n)]


testing(sync_testing, 5)
testing(async_testing, 5)