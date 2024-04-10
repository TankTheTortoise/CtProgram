def mse(y, y_hat, d=False):
    if len(y) != len(y_hat):
        raise Exception("Y and Y_hat aren't the same length")
    else:
        sum = 0
        for i in range(len(y)):
            sum += (y[i] - y_hat[i]) ** 2
        sum /= len(y)
        return sum
        if not d:
            sum = 0
            for i in range(len(y)):
                sum += (y[i] - y_hat[i]) ** 2
            sum /= len(y)
            return sum
        if d:
            sum = 0
            for i in range(len(y)):
                sum += y[i] - y_hat[i]

            sum /= len(y)
            sum *= 2
            return sum


def forward(X, m, b):
    for i in range(len(X)):
        X[i] *= m + b
    return X


data1 = [1, 2, 3, 4]
data2 = [2, 4, 6, 8]

weight = 1
bias = 1

epochs = 1
lr = 1
batch_size = 4

# Data Seperated into minibatches
data1_batched = []
for i in range(0, len(data1), batch_size):
    data1_batched.append(data1[i:i + batch_size])
print(data1_batched)
for epoch in range(epochs):
    train_y = forward(data1, weight, bias)
    for data in data1_batched:
        w_error = 0
        b_error = 0
        for i in range(len(data)):
            w_error -= data1[i] * mse(data2[i], train_y[i], d=True)
            b_error -= mse(data2[i], train_y[i], d=True)
        weight -= w_error / batch_size
        bias -= b_error / batch_size

print(mse(data1, data2))
print(mse(data1, data2, d=True))