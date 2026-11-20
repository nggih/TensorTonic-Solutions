import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here
    upper = float(np.dot(a,b))
    lower = float(np.linalg.norm(a) * np.linalg.norm(b))
    # try:
    #     return float(upper / lower)
    # except Exception as e:
    #     return 0.0
    if lower == 0:
        return 0.0
    return float(upper / lower)
