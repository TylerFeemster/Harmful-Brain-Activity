import numpy as np

@staticmethod
def derivative(y, hz):
    return [(y[i+1]-y[i]) * hz for i in range(len(y)-1)]

class Hjorth:
    def __init__(self, y, hertz):
        self.hz = hertz
        self.y = y
        self.dy = derivative(self.y, hertz)
        self.ddy = derivative(self.dy, hertz)

    def activity(self):
        return np.var(self.y)
    
    def mobility(self):
        var_dy = self.derivative().activity()
        var_y = self.activity()
        return np.sqrt(var_dy / var_y)
    
    def complexity(self):
        dy = self.derivative()
        return dy.mobility() / self.mobility()
    
    def amc(self):
        activity = np.var(self.y)
        d_activity = np.var(self.dy)
        dd_activity = np.var(self.ddy)

        mobility = np.sqrt(d_activity / activity)
        complexity = np.sqrt(dd_activity * activity) / d_activity

        return activity, mobility, complexity