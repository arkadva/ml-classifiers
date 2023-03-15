from modules.models.knn import KNNClassifier
from modules.models.naivebayes import BayesClassifier

from modules.point import Point
from modules.space import Space

import random 

def transform_data(space, sample_space, model_data, ratio=0.8):
    space.clear_space()
    sample_space.clear_space()
    
    with open(model_data) as file:
        lines = [line.rstrip() for line in file]
        line_number = 0
        random_exlusion = random.sample(range(2, len(lines) + 1), int((1 - ratio) * (len(lines) - 1)))
        
        for line in lines:
            line_number += 1
            if line_number == 1:
                continue

            location, date, precipitation, temp_max, temp_min, wind, classifier = line.replace(' ', '').split(',')
            point = Point([float(precipitation), float(temp_max), float(temp_min), float(wind)], classifier)
            
            if line_number in random_exlusion:
                sample_space.add_point(point)
            else:
                space.add_point(point)


def evaluate(sample_space, classifier, param1=10, param2=True):
    total = 0
    correct_predictions = 0
    for point in sample_space.points:
        total += 1
        if classifier.classify(point, param1, param2) == point.classifier:
            correct_predictions += 1
    return correct_predictions/total*100


def main():
    space, sample_space = Space(), Space()
    
    knn = KNNClassifier(space)
    bayes = BayesClassifier(space)
    
    runs = 10
    knn_sum, bayes_sum = 0, 0
    
    for i in range(runs):
        transform_data(space, sample_space, "data/seattle-weather.csv", 0.5)
        res_bayes, res_knn = evaluate(sample_space, bayes), evaluate(sample_space, knn)
        
        knn_sum += res_knn
        bayes_sum += res_bayes
        
        print(f'Bayes {res_bayes}%, KNN {res_knn}%')
      
    print(f'Bayes average {bayes_sum/runs}%')
    print(f'KNN {knn_sum/runs}%')

if __name__ == "__main__":
    main()