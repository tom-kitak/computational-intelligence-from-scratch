import numpy as np


def getFeatures():
    # Open the file "features.txt" located in the data directory
    with open("../data/features.txt") as file:
        # Read each line of the file and convert it to a NumPy array
        features = []
        for line in file:
            # Remove any newline characters and split the line into a list of numbers
            numbers = line.strip().split(",")
            # Convert each number to a float and create a 10x1 NumPy array
            array = np.array([float(num) for num in numbers]).reshape((10, 1))
            # Add the array to the list of features
            features.append(array)

    return features

def getTargetVectors():
    # Open the file "targets.txt" located in the data directory
    with open("../data/targets.txt") as file:
        # Read each line of the file and vectorize the target label
        targets = []
        for line in file:
            # Convert the target label to an integer
            y = int(line.strip())
            # Vectorize the target label using one-hot encoding
            v_y = np.zeros((7, 1))
            v_y[y - 1] = 1.0
            # Add the vectorized target to the list of targets
            targets.append(v_y)

    return targets


def getUnkowns():
    # Open the file "unknown.txt" located in the parent directory
    with open("../data/unknown.txt") as file:
        # Read each line of the file and convert it to a NumPy array
        unknowns = []
        for line in file:
            # Remove any newline characters and split the line into a list of numbers
            numbers = line.strip().split(",")
            # Convert each number to a float and create a 10x1 NumPy array
            array = np.array([float(num) for num in numbers]).reshape((10, 1))
            # Add the array to the list of unknown data
            unknowns.append(array)

    return unknowns

def split_train_test(data, test_size, randomised=False):
    # Splits data into train and test data

    # Determine the index to split the data
    split_index = int(len(data) * (1 - test_size))

    if randomised:
        # Shuffle the data randomly
        np.random.shuffle(data)

    # Split the data into training and testing sets
    train_data = data[:split_index]
    test_data = data[split_index:]

    return train_data, test_data