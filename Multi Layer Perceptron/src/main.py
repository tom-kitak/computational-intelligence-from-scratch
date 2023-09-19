"""
Main module.
"""
from customDataReader import *
from NeuralNetwork import ANN
# import matplotlib
import numpy as np
import Validation
import matplotlib.pyplot as plt
# INFO: data folder must be in the same directory containing input data

def main():
    """
    Main method to run.
    """
    features = getFeatures()
    targets = getTargetVectors()
    unknown = getUnkowns()

    data = list(zip(features, targets))
    training_data, test_data = split_train_test(data, 0.10)
    training_data, validation_data = split_train_test(training_data, 0.10)

    # Create the network with one hidden layer
    network = ANN([10, 30, 7])
    # set to the data to which we are learning to 'data' because we are testing
    # it on the unknown dataset and start training
    
    # Run the ANN normally
    network.train(training_data, validation_data=validation_data)

    # Run the ANN with plotting valdation data agains test data by epoch
    # network.train(train_data=training_data, validation_data=validation_data, test_data=test_data, plot_performance=True)

    accuracy = network.evaluate(test_data)
    print(f"accuracy= {accuracy}")

    # Validation.cross_validation(training_data, [5, 7, 14, 20, 30, 60])


    # # 1.3.10. Train your network 10 times, each with different initial weights
    # accuracies_with_different_initial_weights = []
    # for i in range(1, 11):
    #     network = ANN([10, 30, 7])
    #     network.train(training_data)
    #     accuracy = network.evaluate(test_data)
    #     accuracies_with_different_initial_weights.append(round(accuracy, 3))
    # # plotting 1.3.10.
    # x = np.arange(1, len(accuracies_with_different_initial_weights)+1)
    # fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True)

    # rects = ax.bar(x, accuracies_with_different_initial_weights)

    # for rect in rects:
    #     height = rect.get_height()
    #     print(height)
    #     ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
    #             '{}'.format(height),
    #             ha='center', va='bottom')

    # ax.set_ylabel('Accuracy')
    # ax.set_title('Training network 10 times, each with different initial weights, 20 epochs')
    # ax.set_xticks(x)
    # ax.set_xticklabels(x)
    # plt.ylim((0.5, 1.1))
    # plt.show()
    # # End of 1.3.10.

    



if __name__ == "__main__":
    main()
