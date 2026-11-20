import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    n = len(A)
    m = len(A[0])
    container = []
    for i in range(0, m):
        # n = 2
        # m = 3
        sub_i = []
        for j in range(0, n):  
            sub_i.append(A[j][i])
            
            
        container.append(sub_i) 
    return np.array(container)
        
        
