from time import time

class Timer():
    
    def start(self):
        self.start = time()
        return self
    
    def stop(self):
        self.stop = time()
        return self
    
    @property
    def delta(self, rounding=2):
        try:
            return round(self.stop - self.start, rounding) 
        except:
            raise ValueError("You must have run `start` and `stop` before computing the delta.")
    
    @property
    def start_time(self):
        try:
            return self.start
        except:
            raise ValueError("Timer has never been started.")
    
    @property
    def start_time(self):
        try:
            return self.stop
        except:
            raise ValueError("Timer has never been stopped.")