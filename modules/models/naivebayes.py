from math import exp, pi, sqrt
noise = 10**(-5)

class BayesClassifier:
    def __init__(self, space):
        self.space = space
    
    def add_mean_var(self):
        self.mean = {}
        self.variance = {}
        self.prior = {}
        self.total = 0
        
        for classification in self.space.classifications:
            self.mean[classification] = [[] for _ in self.space.points[0].dimensions]
            self.variance[classification] = self.mean[classification].copy()
            self.prior[classification] = 0
        
        for point in self.space.points:
            for dimension in range(len(point.dimensions)):
                self.mean[point.classifier][dimension].append(point.dimensions[dimension])
            self.prior[point.classifier] += 1
            self.total += 1
        
        for classification in self.space.classifications:
            for dimension in range(len(self.mean[point.classifier])):
                self.mean[classification][dimension], self.variance[classification][dimension] = BayesClassifier.calc_mean_var(self.mean[classification][dimension])
    
    def classify(self, data, param1=None, param2=None):
        self.add_mean_var()
        data=data.dimensions
        
        best_classification = None
        best_p = 0
        
        for classification in self.space.classifications:
            cumulative_p = self.prior[classification]/self.total

            for dimension in range(len(data)):
                variance = self.variance[classification][dimension] + noise
                mean = self.mean[classification][dimension]

                p = 1/(sqrt(2*pi*variance))*exp(-((data[dimension] - mean)**2)/(2*variance))
                cumulative_p *= p
                
            if cumulative_p > best_p:
                best_p = cumulative_p
                best_classification = classification
                
            cumulative_p = self.prior[classification]/self.total
        return best_classification
        
    @staticmethod
    def calc_mean_var(data):
        mean = sum(data)/len(data)
        variance = sum(map(lambda x: pow(x - mean, 2), data))/(len(data) - 1)
        return mean, variance