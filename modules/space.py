class Space:
    def __init__(self):
        self.points = []
        self.classifications = []
        
    def add_point(self, point):
        self.points.append(point)
        if point.classifier not in self.classifications:
            self.classifications.append(point.classifier)
    
    def clear_space(self):
        self.points.clear()
        self.classifications.clear()