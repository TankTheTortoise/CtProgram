def mse(y, y_hat, d=False):
    if not d:
        return (y_hat-y) ** 2
    if d:
        return 2 * (y_hat - y)


def forward(x, m, b):
    x = x * m + b
    return x


def element_mul(list1, scalar):
    for i in range(len(list1)):
        list1[i] *= scalar
    return list1[i]


data1 = [1, 2, 3, 4]
data2 = [2, 4, 6, 8]

weight = 2
bias = 0.01

epochs = 1000
lr = 0.001

batch_size=1
batched_data1 = []
batched_data2 = []
for d in range(batch_size, len(data1)+1):
    batched_data1.append(data1[d-batch_size: d])
for d in range(batch_size, len(data2)+1):
    batched_data2.append(data2[d-batch_size: d])

print(batched_data1)
samples = len(data1)
for epoch in range(epochs):
    for i in range(0, samples):
        output = 0
        error = 0
        for j in range(batch_size):
            output += forward(batched_data1[i][j], weight, bias)
            error += mse(batched_data2[i][j], output, d=True)
        weight -= (error * data1[i] * lr)
        bias -= (error * lr)/batch_size
print(weight, bias)

print(weight, bias)
for j in range(len(data1)):
    data1[j] = data1[j] * weight + bias
print(data1)
