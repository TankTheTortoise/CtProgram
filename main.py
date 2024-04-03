def mse(y, y_hat):
    if len(y) != len(y_hat):
        raise Exception("Y and Y_hat aren't the same length")
    else:
        sum = 0
        for i in range(len(y)):
            sum += (y[i] - y_hat[i]) ** 2
        sum /= len(y)
        return sum
