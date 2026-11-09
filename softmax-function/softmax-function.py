import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    # Write code here
    arr = np.array(x)
    print(arr.shape)
    if arr.ndim == 2:
        max_reduced_arr = arr - np.max(arr, axis=1, keepdims=True)
        exp = np.exp(max_reduced_arr)
        return exp / np.sum(exp, axis=1, keepdims=True)
    else:
        max_reduced_arr = arr - np.max(arr)
        exp = np.exp(max_reduced_arr)

        return exp / np.sum(exp)