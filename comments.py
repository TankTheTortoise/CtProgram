# Returns the mean squared error of an answer and a guess
def mse(y, y_hat, d=False):
    if len(y) != len(y_hat):
        raise Exception("Lengths of inputted lists aren't the same")
    error = 0
    for i in range(len(y)):
        if not d:
            error += (y_hat[i] - y[i]) ** 2
        if d:
            error += 2 * (y_hat[i] - y[i])
    error /= len(y)
    return error


# These all open a file, read the file into a string, and then split the string by commas into a list of numbers

# Input data for testing
with open("testX.txt", "r") as testX_file:
    testX = testX_file.readline()
testX = testX.split(",")
for i in range(len(testX)):
    testX[i] = float(testX[i])
# Answers for testing
with open("testY.txt", "r") as testY_file:
    testY = testY_file.readline()
testY = testY.split(",")
for i in range(len(testY)):
    testY[i] = float(testY[i])
# Input data for training
with open("trainX.txt", "r") as trainX_file:
    trainX = trainX_file.readline()
trainX = trainX.split(",")
for i in range(len(trainX)):
    trainX[i] = float(trainX[i])
# Answers for training
with open("trainY.txt", "r") as trainY_file:
    trainY = trainY_file.readline()
trainY = trainY.split(",")
for i in range(len(trainY)):
    trainY[i] = float(trainY[i])

# The weight is multiplied by the incoming data. It is assigned a random value at the beginning
weight:float = 0.5
# The bias is added by the incoming data * weight. It is assigned a random value at the beginning
bias = 0.01

# Epochs are how many times the entire set of data is looped through
epochs = 10
# The learning rate adjusts how much the model weights the errors.
lr = 0.01

samples = len(trainX)
for epoch in range(epochs):
    for i in range(0, samples):
        output = 0
        error = 0
        w_error = 0
        output += trainX[i] * weight + bias
        error += mse([trainY[i]], [output], d=True)
        w_error += mse([trainY[i]], [output], d=True) * trainX[i]

        weight -= lr * w_error
        bias -= lr * error

# The weights and biases
weight = round(weight, 5)
bias = round(bias, 5)

# Calculates the mean squared error
test_outputs = [j * weight + bias for j in testX]
test_outputs = mse(testY, test_outputs)
print(f"Mean squared error: {test_outputs}")
