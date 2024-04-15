def mse(y, y_hat, d=False):
    if not d:
        return (y_hat-y) ** 2
    if d:
        return 2 * (y_hat - y)


def forward(x, m, b):
    x = x * m + b
    return x


data1 = [x for x in range(4)]
data2 = [x*2+3 for x in range(4)]

weight = 2
bias = 0.01

epochs = 1000
lr = 0.01

batch_size=2
batched_data1 = []
batched_data2 = []
for d in range(batch_size, len(data1)+1):
    batched_data1.append(data1[d-batch_size: d])
for d in range(batch_size, len(data2)+1):
    batched_data2.append(data2[d-batch_size: d])

print(batched_data1)
samples = len(batched_data1)
for epoch in range(epochs):
    for i in range(0, samples):
        output = 0
        error = 0
        for j in range(0, int(len(data1)/batch_size)-1):
            output += forward(batched_data1[i][j], weight, bias)
            error += mse(batched_data2[i][j], output, d=True)
        weight -= (error * data1[i] * lr)
        bias -= (error * lr)/batch_size

    print(weight, bias)

weight = round(weight, 4)
bias = round(bias, 4)
for j in range(len(data1)):
    data1[j] = round(data1[j] * weight + bias, 4)
print(data1)
