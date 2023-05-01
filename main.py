from modules.models.knn import KNNClassifier
from modules.models.naivebayes import BayesClassifier

from modules.point import Point
from modules.space import Space

import random 

def transform_data(space, sample_space, model_data, transform, ratio):
    space.clear_space()
    sample_space.clear_space()
    
    with open(model_data) as file:
        data = [transform(line.rstrip().replace(', ', ',').split(',')) for i, line in enumerate(file) if i > 0]
        
        random.shuffle(data)
        split_index = int(len(data) * ratio)
        
        for point in data[:split_index]:
            space.add_point(point)
        
        for point in data[split_index:]:
            sample_space.add_point(point)

def evaluate(sample_space, classifier, param1=10, param2=True):
    correct_predictions = 0
    for point in sample_space.points:
        if classifier.classify(point, param1, param2) == point.classifier:
            correct_predictions += 1
    return correct_predictions/len(sample_space.points)


def main():
    # adjust your parameters here
    runs = 10
    split_ratio = 0.5
    file_name = "data/seattle-weather.csv"
    
    def transform(line):
        """
        This function transforms a list of items from a dataset into a Point object. 
        The function assumes that the input data is in a specific order, and therefore 
        extracts the necessary values based on their positions in the input list. If the input 
        data needs to be normalized, this function can be modified to include normalization 
        steps. 

        Arguments:
        line - A list of items containing data for a single observation. 

        Returns:
        A Point object representing the input data and a class label for the observation. The class label
        type agnostic.
        """
        location, date, precipitation, temp_max, temp_min, wind, classifier = line
        point = Point([float(precipitation), float(temp_max), float(temp_min), float(wind)], classifier)
        return point

    space, sample_space = Space(), Space()
    
    knn = KNNClassifier(space)
    bayes = BayesClassifier(space)
    
    knn_sum, bayes_sum = 0, 0
    
    for i in range(runs):
        transform_data(space, sample_space, file_name, transform, split_ratio)
        res_bayes, res_knn = evaluate(sample_space, bayes), evaluate(sample_space, knn)
        
        knn_sum += res_knn
        bayes_sum += res_bayes
        
        print(f'GN Bayes {res_bayes}, KNN {res_knn}')
      
    print(f'GN Bayes average {bayes_sum/runs}')
    print(f'KNN {knn_sum/runs}')

if __name__ == "__main__":
    main()