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

# 1
test_pattern = [1, -1, -1, -1, -1]
print(f"Input Pattern: {test_pattern}\n")

predicted_sync = hnet.predict(test_pattern, mode='sync')
print(f"Predicted Pattern (sync): {predicted_sync}\n")

predicted_async = hnet.predict(test_pattern, mode='async')
print(f"Predicted Pattern (async): {predicted_async}\n")

# 2
test_pattern = [1, -1, -1, -1, -1]
print(f"Input Pattern: {test_pattern}\n")

predicted_sync = hnet.predict(test_pattern, mode='sync')
print(f"Predicted Pattern (sync): {predicted_sync}\n")

# 3
test_pattern = [1, 1, -1, -1, -1]
print(f"Input Pattern: {test_pattern}\n")

predicted_sync = hnet.predict(test_pattern, mode='sync')
print(f"Predicted Pattern (sync): {predicted_sync}\n")