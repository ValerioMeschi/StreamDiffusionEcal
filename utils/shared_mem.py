
from multiprocessing import shared_memory
import numpy as np

def create_shared_float(initial_value=0.0):
    # Create a numpy array with a single float value
    arr = np.array([initial_value], dtype=np.float64)

    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)

    # Create a numpy array that uses the shared memory
    shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shared_arr[:] = arr[:]  # Copy the initial value

    return shm.name, shm

# Access the shared float
def access_shared_float(shm_name):
    # Attach to existing shared memory
    shm = shared_memory.SharedMemory(name=shm_name)

    # Create a numpy array using the shared memory buffer
    shared_arr = np.ndarray((1,), dtype=np.float64, buffer=shm.buf)
    return shared_arr, shm

# Clean up shared memory
def cleanup_shared_float(shm_name):
    arr, shm = access_shared_float(shm_name)
    shm.close()
    shm.unlink()
