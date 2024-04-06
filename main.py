def mse(y, y_hat, d=False):
    if len(y) != len(y_hat):
        raise Exception("Y and Y_hat aren't the same length")
    else:
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


def forward(X, m):
    for i in range(X):
        X[i] *= m


data1 = [1, 2, 3]
data2 = [4, 5, 6]

print(mse(data1, data2))
print(mse(data1, data2, d=True))
