import numpy as np
from tabulate import tabulate
from hopfield_network import HopfieldNetwork

network_size = 5
hnet = HopfieldNetwork()

patterns = [
    [1, -1, 1, -1, 1],
    [-1, 1, -1, 1, -1]
]

hnet.train(np.array(patterns))
print(f"Training Completed.\n{tabulate(patterns, tablefmt="fancy_grid")}")


def pattern_gen(n=network_size):
    return np.random.choice([-1, 1], size=(n, n))


def sync_testing(test_pattern=None):
    if (test_pattern is None):
        test_pattern = pattern_gen()

    print(f"Input Pattern:\n{tabulate(test_pattern, tablefmt="grid")}\n")
    
    predicted_sync = hnet.predict(test_pattern, mode='sync')
    print(f"\nTrained Pattern:\n{tabulate(patterns, tablefmt="heavy_grid")}\n")
    print(f"Predicted Pattern (sync): \n{tabulate(predicted_sync, tablefmt="fancy_grid")}\n\n")


def async_testing(test_pattern=None):
    if (test_pattern is None):
        test_pattern = pattern_gen()

    print(f"Input Pattern:\n{tabulate(test_pattern, tablefmt="grid")}\n")
    
    predicted_async = hnet.predict(test_pattern, mode='async')
    print(f"\nTrained Pattern:\n{tabulate(patterns, tablefmt="heavy_grid")}\n")
    print(f"Predicted Pattern (sync): \n{tabulate(predicted_async, tablefmt="fancy_grid")}\n\n")



def testing(fn, n):
    [fn() for _ in range(n)]


testing(sync_testing, 5)
testing(async_testing, 5)
