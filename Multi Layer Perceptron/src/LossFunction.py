import numpy as np

class MeanSquaredError:

    @staticmethod
    def loss(output_prediction, true_label):
        return (output_prediction - true_label) ** 2
    
    @staticmethod
    def derivative(output_prediction, true_label):
        return 2 * (output_prediction - true_label)
    

class CategoricalCrossEntropy:

    @staticmethod
    def loss(output_prediction, true_label):
        return -np.sum(true_label * np.log(output_prediction))
    
    @staticmethod
    def derivative(output_prediction, true_label):
        return -true_label/(output_prediction)

       