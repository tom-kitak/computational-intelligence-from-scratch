import numpy as np

class Softmax:

    @staticmethod
    def activation(x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)
    
    @staticmethod
    def derivative(x):
        # calculations based on: https://www.youtube.com/watch?v=gRr2Q97XS2g
        # and also: https://neuralthreads.medium.com/softmax-function-it-is-frustrating-that-everyone-talks-about-it-but-very-few-talk-about-its-54c90b9d0acd
        I = np.eye(x.shape[0])
        jacobian_m = Sigmoid.activation(x) * (I - Sigmoid.activation(x).T)
        #target = np.argmax(target_for_derivative)
        #return np.array(jacobian_m[:, target]).reshape(len(x), -1)
        return jacobian_m

        
    
class Sigmoid:

    @staticmethod
    def activation(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def derivative(x):
        return (np.exp(-x)) / ((np.exp(-x) + 1)**2)
    

class Tanh:
    
    @staticmethod
    def activation(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


    @staticmethod
    def derivative(x):
        return 1 - Tanh.activation(x) ** 2
        
    