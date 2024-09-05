import random
from collections import deque, namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class MemoryStack(object):
    """
    This class creates a MemoryStack object with size `capacity`,
    upon reaching capacity the oldest objects will be dropped.

    Attributes:
        memory (deque([], maxlen=capacity)): The memory deque which stores the saved objects -- typically transition tensors.
    """
    def __init__(self, capacity):
        """
        The constructor for the MemoryStack class.

        Parameters:
            capacity (int): The number of objects which will be stored before the oldest objects start being dropped from the deque stack.
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, x):
        """
        Push an object to the MemoryStack `.memory` deque.

        Parameters:
            x (any): The element to push to the MemoryStack.
        """
        self.memory.append(x)
    
    def sample(self, batch_size):
        """
        Pull `batch_size` random samples from the MemoryStack `.memory` deque.

        Parameters:
            batch_size (int): The number of individual samples to pull from the memory.
        
        Returns:
            array: An array containing `batch_size` individual samples from memory.
        """
        return random.sample(self.memory, batch_size)