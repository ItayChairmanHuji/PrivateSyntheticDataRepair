import numpy as np
class MarginalsDistance: 
    def __call__(self, private_count: int, synthetic_count: int) -> float:
        return np.abs(private_count - synthetic_count)