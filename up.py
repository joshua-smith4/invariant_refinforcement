import numpy as np
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.random.set_seed(42)
np.random.seed(42)

train_x = []
train_y = []

train_x0 = []
train_y0 = []

train_x1 = []
train_y1 = []

num_classes = 4
points_per_class = 100
batch_size = 20
num_epochs = 80

for i in range(num_classes):
    for j in range(points_per_class):
        tmp = np.random.rand(2, 1).astype(np.float32) / 2
        if i == 0:
            pass
            train_y.append(0.0)
            train_y0.append(0.0)
            train_x0.append(tmp)
        elif i == 1:
            tmp[0] += 0.5
            train_y.append(1.0)
            train_y1.append(1.0)
            train_x1.append(tmp)
        elif i == 2:
            tmp[1] += 0.5
            train_y.append(1.0)
            train_y1.append(1.0)
            train_x1.append(tmp)
        elif i == 3:
            tmp += 0.5
            train_y.append(0.0)
            train_y0.append(0.0)
            train_x0.append(tmp)
        train_x.append(tmp)


dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))

i = tf.Variable(train_x[0], dtype=tf.float32, trainable=False)
l = tf.Variable(train_y[0], dtype=tf.float32, trainable=False)

w1 = tf.Variable(tf.random.normal((5, 2)), dtype=tf.float32)
b1 = tf.Variable(tf.random.normal((5, 1)), dtype=tf.float32)

w2 = tf.Variable(tf.random.normal((3, 5)), dtype=tf.float32)
b2 = tf.Variable(tf.random.normal((3, 1)), dtype=tf.float32)

w3 = tf.Variable(tf.random.normal((1, 3)), dtype=tf.float32)
b3 = tf.Variable(tf.random.normal((1, 1)), dtype=tf.float32)

trainable_parameters = [w1, b1, w2, b2, w3, b3]


def model():
    h1 = tf.nn.relu(tf.matmul(w1, i) + b1)
    h2 = tf.nn.relu(tf.matmul(w2, h1) + b2)
    h3 = tf.nn.sigmoid(tf.matmul(w3, h2) + b3)
    return h1, h2, h3


optimizer = tf.keras.optimizers.Adam()


def train_step(inputs, labels, alpha=0.001):
    acc_grads = [tf.zeros_like(x) for x in trainable_parameters]
    for p in range(len(inputs)):
        x = inputs[p]
        y = labels[p]
        with tf.GradientTape(persistent=True) as tape:
            i.assign(x)
            l.assign(y)
            h1, h2, h3 = model()
            loss = tf.keras.losses.mean_squared_error(h3, l)
        grads_h1 = tape.gradient(h1, trainable_parameters)
        grads_h2 = tape.gradient(h2, trainable_parameters)
        grads_h3 = tape.gradient(h3, trainable_parameters)
        grads_loss = tape.gradient(loss, trainable_parameters)
        del tape
        for j in range(len(grads_loss)):
            acc_grads[j] += grads_loss[j]
    for weight, grad in zip(trainable_parameters, acc_grads):
        weight.assign(weight - alpha * grad)


def predicate(val, label):
    return (val >= 0.5 and label >= 0.5) or (val < 0.5 and label < 0.5)

def accuracy(inputs, labels):
    num_correct = 0
    for x, y in zip(inputs, labels):
        i.assign(x)
        _, _, h3 = model()
        val = h3.numpy()[0, 0]
        if predicate(val, y):
            num_correct += 1
    return num_correct / len(inputs)


def success_variance(inputs, label):
    points_h1 = []
    points_h2 = []
    points_h3 = []
    for x in inputs:
        i.assign(x)
        h1, h2, h3 = model()
        val = h3.numpy()[0,0]
        if predicate(val, label):
            points_h1.append(h1.numpy())
            points_h2.append(h2.numpy())
            points_h3.append(h3.numpy())
    return np.var(points_h1, axis=0), np.var(points_h2, axis=0), np.var(points_h3, axis=0)

def success_stddev(inputs, label):
    points_h1 = []
    points_h2 = []
    points_h3 = []
    for x in inputs:
        i.assign(x)
        h1, h2, h3 = model()
        val = h3.numpy()[0,0]
        if predicate(val, label):
            points_h1.append(h1.numpy())
            points_h2.append(h2.numpy())
            points_h3.append(h3.numpy())
    return np.std(points_h1, axis=0), np.std(points_h2, axis=0), np.std(points_h3, axis=0)

dataset = dataset.shuffle(len(dataset))
dataset = dataset.batch(40)

h03_var = []
h13_var = []
h03_std = []
h13_std = []

for epoch in range(num_epochs):
    print(f'Epoch: {epoch}')
    for x, y in dataset:
        train_step(x, y, 0.01)
    v01, v02, v03 = success_variance(train_x0, 0)
    v11, v12, v13 = success_variance(train_x1, 1)
    s01, s02, s03 = success_stddev(train_x0, 0)
    s11, s12, s13 = success_stddev(train_x1, 1)
    h03_var.append(v03)
    h13_var.append(v13)
    h03_std.append(s03)
    h13_std.append(s13)
    acc = accuracy(train_x, train_y)
    per = acc * 100
    print(f"Accuracy: {per}")
