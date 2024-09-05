import time

class TimeTracker():
    """
    Tracks Time

    Attributes:
        title (str): The title of the time tracker.
        start_time (int): The starting time of the counter, in nanoseconds since the Epoch.
        end_time (int): The ending time of the counter, in nanoseconds since the Epoch.
    """
    def __init__(self, title):
        """
        The constructor for the TimeTracker class.

        Parameters:
            title (str): The title of the time tracker.
        """
        self.title = title
        self.start_time = time.time_ns()
        self.end_time = time.time_ns()

    def start(self):
        """
        Sets the start time of the time tracker.
        """
        self.start_time = time.time_ns()
        return self.start_time
    
    def end(self):
        """
        Sets the end time of the time tracker.
        """
        self.end_time = time.time_ns()
        return self.end_time
    
    def print_log(self):
        """
        Sets the end of the timer, and then prints the difference between start_time and end_time (the total time elapsed).
        """
        self.end()
        delta = (self.end_time - self.start_time)

        print(f"{self.title} took {round(delta * 1e-6, 0)}ms")