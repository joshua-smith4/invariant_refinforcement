import numpy as np
import tensorflow as tf
import pickle
import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument('--inv_reinf', default='false')
parser.add_argument('--out_file', default='out.pickle')
parser.add_argument('--normalize_batch', default='true')
parser.add_argument('--normalize_neurons', default='true')
parser.add_argument('--min_layer', type=int, default=0)
parser.add_argument('--func', default='xor')
parser.add_argument('--save_file', default="network.pickle")

args = parser.parse_args()
args.inv_reinf = args.inv_reinf == 'true'
if args.inv_reinf:
    print(
        f'using invariant reinforcement on layers greater than or equal to {args.min_layer}')
args.normalize_batch = args.normalize_batch == 'true'
if args.normalize_batch:
    print('normalizing by size of batch')
args.normalize_neurons = args.normalize_neurons == 'true'
if args.normalize_neurons:
    print('normalizing by number of neurons')

tf.random.set_seed(42)
np.random.seed(42)

train_x = []
train_y = []

train_x0 = []
train_y0 = []

train_x1 = []
train_y1 = []

num_classes = 2
num_cases = 4
points_per_case = 100
batch_size = 20
num_epochs = 80

if args.func == 'xor':
    print('training xor function')
elif args.func == 'and':
    print('training and function')

for i in range(num_cases):
    for j in range(points_per_case):
        tmp = np.random.rand(2, 1).astype(np.float32) / 2
        if i == 0:
            train_y.append(0.0)
            train_y0.append(0.0)
            train_x0.append(tmp)
        elif i == 1:
            tmp[0] += 0.5
            if args.func == 'xor':
                train_y.append(1.0)
                train_y1.append(1.0)
                train_x1.append(tmp)
            if args.func == 'and':
                train_y.append(0.0)
                train_y0.append(0.0)
                train_x0.append(tmp)
        elif i == 2:
            tmp[1] += 0.5
            if args.func == 'xor':
                train_y.append(1.0)
                train_y1.append(1.0)
                train_x1.append(tmp)
            if args.func == 'and':
                train_y.append(0.0)
                train_y0.append(0.0)
                train_x0.append(tmp)
        elif i == 3:
            tmp += 0.5
            if args.func == 'xor':
                train_y.append(0.0)
                train_y0.append(0.0)
                train_x0.append(tmp)
            if args.func == 'and':
                train_y.append(1.0)
                train_y1.append(1.0)
                train_x1.append(tmp)
        train_x.append(tmp)


dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))

i = tf.Variable(train_x[0], dtype=tf.float32, trainable=False)
l = tf.Variable(train_y[0], dtype=tf.float32, trainable=False)

w1 = tf.Variable(tf.random.normal((5, 2)), dtype=tf.float32)
b1 = tf.Variable(tf.random.normal((5, 1)), dtype=tf.float32)

w2 = tf.Variable(tf.random.normal((5, 5)), dtype=tf.float32)
b2 = tf.Variable(tf.random.normal((5, 1)), dtype=tf.float32)

w3 = tf.Variable(tf.random.normal((5, 5)), dtype=tf.float32)
b3 = tf.Variable(tf.random.normal((5, 1)), dtype=tf.float32)

w4 = tf.Variable(tf.random.normal((1, 5)), dtype=tf.float32)
b4 = tf.Variable(tf.random.normal((1, 1)), dtype=tf.float32)

trainable_parameters = [w1, b1, w2, b2, w3, b3, w4, b4]


def model():
    h1 = tf.nn.relu(tf.matmul(w1, i) + b1)
    h2 = tf.nn.relu(tf.matmul(w2, h1) + b2)
    h3 = tf.nn.relu(tf.matmul(w3, h2) + b3)
    h4 = tf.nn.sigmoid(tf.matmul(w4, h3) + b4)
    neurons = [[x for x in h1], [x for x in h2],
               [x for x in h3], [x for x in h4]]
    return h1, h2, h3, h4, neurons


def v(l, _i, x):
    i.assign(x)
    h1, h2, h3, h4, _ = model()
    layers = [h1, h2, h3, h4]
    layer = layers[l].numpy()
    print(layer)
    return layer[_i][0]


optimizer = tf.keras.optimizers.Adam()


def predicate(val, label):
    return (val >= 0.5 and label >= 0.5) or (val < 0.5 and label < 0.5)


def train_step(inputs, labels, inv_reinforce, alpha, safe_neurons, safe_means, safe_variances, acc_threshold, num_training_points):
    acc_grads = [tf.zeros_like(x) for x in trainable_parameters]
    inv_reinf_grads = [tf.zeros_like(x) for x in trainable_parameters]
    num_in_batch_applied = 0
    num_neurons_applied = 0
    num_neurons = None
    for p in range(len(inputs)):
        x = inputs[p]
        y = labels[p]
        with tf.GradientTape(persistent=True) as tape:
            i.assign(x)
            l.assign(y)
            h1, h2, h3, h4, neurons = model()
            loss = tf.keras.losses.mean_squared_error(h4, l)
        grads_loss = tape.gradient(loss, trainable_parameters)
        for j in range(len(grads_loss)):
            acc_grads[j] += grads_loss[j]
        classification = h4[0][0].numpy()
        label = y.numpy()
        if num_neurons is None:
            num_neurons = sum([len(layer) for layer in neurons])
        class_percent_safe = safe_neurons[int(
            label)][0].shape[0] / num_training_points
        if not inv_reinforce or not predicate(classification, label) or class_percent_safe < acc_threshold:
            del tape
            continue
        num_in_batch_applied += 1
        for layer in range(len(neurons)-1, args.min_layer - 1, -1):
            num_neurons_applied += len(neurons[layer])
            for neuron_index, neuron in enumerate(neurons[layer]):
                m = safe_means[int(label)][layer][neuron_index][0]
                v = safe_variances[int(label)][layer][neuron_index][0]
                neuron_grad = tape.gradient(neuron, trainable_parameters)
                s = 0.0
                if v > 0:
                    s = (tf.nn.sigmoid((m - neuron.numpy())/np.sqrt(v)).numpy()
                         [0] - 0.5) * 2.0
                for grad_index, cur_grad in enumerate(neuron_grad):
                    if cur_grad is None:
                        continue
                    inv_reinf_grads[grad_index] += s * cur_grad
        del tape

    for weight, grad, inv_grad in zip(trainable_parameters, acc_grads, inv_reinf_grads):
        sgd_update = -alpha*grad/len(inputs)
        inv_update = np.zeros_like(weight)
        if num_in_batch_applied > 0:
            inv_update = alpha * inv_grad
            if args.normalize_batch:
                inv_update /= num_in_batch_applied
            if args.normalize_neurons:
                inv_update /= num_neurons_applied
        weight.assign(weight + sgd_update + inv_update)


def accuracy(inputs, labels):
    num_correct = 0
    for x, y in zip(inputs, labels):
        i.assign(x)
        _, _, _, h4, _ = model()
        val = h4.numpy()[0][0]
        if predicate(val, y):
            num_correct += 1
    return num_correct / len(inputs)


def allSafeNeuronValuesOverClass(C, label):
    layer0 = []
    layer1 = []
    layer2 = []
    layer3 = []
    for c in C:
        i.assign(c)
        h1, h2, h3, h4, _ = model()
        val = h4.numpy()[0][0]
        if not predicate(val, label):
            continue
        layer0 += [h1]
        layer1 += [h2]
        layer2 += [h3]
        layer3 += [h4]
    return [np.array(layer0), np.array(layer1), np.array(layer2), np.array(layer3)]


def calcMeanAndVariance(vals):
    means = []
    variances = []
    for v in vals:
        means += [np.mean(v, axis=0)]
        variances += [np.var(v, axis=0)]
    return means, variances


dataset = dataset.shuffle(len(dataset))
dataset = dataset.batch(10)


observed_neurons = [(0, 0), (1, 0), (2, 0), (3, 0),
                    (0, 1), (0, 2), (0, 3), (0, 4),
                    (1, 1), (1, 2), (1, 3), (1, 4),
                    (2, 1), (2, 2), (2, 3), (2, 4)]

observed_means0 = [[] for x in observed_neurons]
observed_means1 = [[] for x in observed_neurons]
observed_variances0 = [[] for x in observed_neurons]
observed_variances1 = [[] for x in observed_neurons]
num_correctly_classified = []
accuracy_hist = []
epoch_acc = []

for epoch in range(num_epochs):
    print(f'Epoch: {epoch}')
    for ind, (x, y) in enumerate(dataset):
        c0 = allSafeNeuronValuesOverClass(train_x0, 0)
        c1 = allSafeNeuronValuesOverClass(train_x1, 1)
        num_correctly_classified.append(len(c0[0]))
        m0, v0 = calcMeanAndVariance(c0)
        m1, v1 = calcMeanAndVariance(c1)
        for index, (_l, _i) in enumerate(observed_neurons):
            m0_val = np.nan
            v0_val = np.nan
            m1_val = np.nan
            v1_val = np.nan
            if len(c0[0]) > 0:
                m0_val = m0[_l][_i][0]
                v0_val = v0[_l][_i][0]
            if len(c1[0]) > 0:
                m1_val = m1[_l][_i][0]
                v1_val = v1[_l][_i][0]
            observed_means0[index].append(m0_val)
            observed_variances0[index].append(v0_val)
            observed_means1[index].append(m1_val)
            observed_variances1[index].append(v1_val)
        acc = accuracy(train_x, train_y)
        print(f'Accuracy {acc}, batch {ind}')
        accuracy_hist.append(acc)
        train_step(x, y, args.inv_reinf, 0.01, [c0, c1], [
                   m0, m1], [v0, v1], 0.3, len(train_x))
    acc = accuracy(train_x, train_y)
    per = acc * 100
    epoch_acc.append(acc)
    print(f'Accuracy {per}')

d = {}
d['var0'] = observed_variances0
d['var1'] = observed_variances1
d['mean0'] = observed_means0
d['mean1'] = observed_means1
d['correct'] = num_correctly_classified
d['acc'] = accuracy_hist
d['epoch_acc'] = epoch_acc
with open(args.out_file, 'wb') as f:
    pickle.dump(d, f)

n = {}
n['w1'] = w1.numpy()
n['b1'] = b1.numpy()
n['w2'] = w2.numpy()
n['b2'] = b2.numpy()
n['w3'] = w3.numpy()
n['b3'] = b3.numpy()
n['w4'] = w4.numpy()
n['b4'] = b4.numpy()
with open(args.save_file, 'wb') as f:
    pickle.dump(n, f)
