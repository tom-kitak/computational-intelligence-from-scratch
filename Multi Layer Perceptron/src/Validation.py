import numpy as np
import math
from NeuralNetwork import ANN

def internal_cross_validation(hyperparameter, data, k_folds=10):

    scores = []

    for fold in range(k_folds):
        start = math.floor((len(data)/k_folds)*fold)
        end = math.floor((len(data)/k_folds)*(fold+1))
        validation_data = data[start:end]
        train_data = data[:start] + data[end:]
        print("fold number:", fold+1)

        k_fold_model = ANN([10, hyperparameter, 7])
        k_fold_model.train(train_data)

        validation_data_error = k_fold_model.evaluate(validation_data)

        scores.append(validation_data_error)

    return np.average(scores)

def plot_hidden_neurons_performance(hidden_neurons):
    import matplotlib.pyplot as plt
    x = np.arange(len(hidden_neurons['hyperparameters']))

    fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True)

    rects = ax.bar(x, hidden_neurons['performances'])
    print(rects)

    for rect in rects:
        height = rect.get_height()
        print(height)
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '{}'.format(height),
                ha='center', va='bottom')

    ax.set_ylabel('K-Fold Cross Validation Performance')
    ax.set_title('Number of Hidden Neurons')
    ax.set_xticks(x)
    ax.set_xticklabels(hidden_neurons['hyperparameters'])
    plt.ylim((0.5, 1.1))

    plt.show()

def cross_validation(training_data, hyperparameters):
    hidden_neurons = {
        "hyperparameters" : hyperparameters,
        "performances" : []
    }

    for parameter in hidden_neurons["hyperparameters"]:
        performance = internal_cross_validation(parameter, training_data)
        print("parameter", parameter, "score is", performance)
        hidden_neurons["performances"].append(round(performance, 3))
    
    plot_hidden_neurons_performance(hidden_neurons)


