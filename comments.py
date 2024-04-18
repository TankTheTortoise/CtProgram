def mse(y, y_hat, d=False):
    if not d:
        return (y_hat - y) ** 2
    if d:
        return 2 * (y_hat - y)


# These all open a file, read the file into a string, and then split the string by commas into a list of numbers

# Training
with open("testX.txt", "r") as testX_file:
    testX = testX_file.readline()
testX = testX.split(",")
for i in range(len(testX)):
    testX[i] = float(testX[i])

with open("testY.txt", "r") as testY_file:
    testY = testY_file.readline()
testY = testY.split(",")
for i in range(len(testY)):
    testY[i] = float(testY[i])

with open("trainX.txt", "r") as trainX_file:
    trainX = trainX_file.readline()
trainX = trainX.split(",")
for i in range(len(trainX)):
    trainX[i] = float(trainX[i])

with open("trainY.txt", "r") as trainY_file:
    trainY = trainY_file.readline()
trainY = trainY.split(",")
for i in range(len(trainY)):
    trainY[i] = float(trainY[i])
print(trainX)
print(trainY)
print(testX)
print(testY)

# The weight is multiplied by the incoming data. It is assigned a random value at the beginning
weight = 20
# The bias is added by the incoming data * weight. It is assigned a random value at the beginning
bias = 0.01

# Epochs are how many times the entire set of data is looped through
epochs = 1000
# The learning rate adjusts how much the model weights the errors.
lr = 0.01

batch_size = 1
samples = len(trainX)
for epoch in range(epochs):
    for i in range(0, samples, batch_size):
        output = 0
        error = 0
        w_error = 0
        for j in range(0, batch_size):
            output += trainX[i + j] * weight + bias
            error += mse(trainY[i + j], output, d=True)
            w_error += mse(trainY[i + j], output, d=True) * trainX[i + j]

        weight -= lr * (w_error/ batch_size)
        bias -= lr * (error/ batch_size)

weight = round(weight, 5)
bias = round(bias, 5)
test_error = 0
# Calculates the mean squared error
for j in range(len(testX)):
    test_error += mse(testY[j], (testX[j] * weight + bias))
    print(testY[j], (testX[j] * weight + bias))
print(f"Mean squared error: {test_error / len(testX)}")
