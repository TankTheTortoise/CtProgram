def mse(y, y_hat, d=False):
    if not d:
        return (y_hat - y) ** 2
    if d:
        return 2 * (y_hat - y)


with open("trainY.txt", "r") as testX_file:
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

with open("testX.txt", "r") as trainY_file:
    trainY = trainY_file.readline()
trainY = trainY.split(",")
for i in range(len(trainY)):
    trainY[i] = float(trainY[i])

weight = 2
bias = 345

epochs = 100
lr = 0.001

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

        weight -= (lr * w_error) / batch_size
        bias -= (error * lr) / batch_size
        print(weight, bias)


for j in range(len(testX)):
    mse(testY[j], (testX[j] * weight + bias) / testY[j])
