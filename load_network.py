import pickle
import argparse
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--network_file', default='network.pickle')

args = parser.parse_args()

with open(args.network_file, 'rb') as f:
    nn = pickle.load(f)

i = tf.Variable(np.zeros((2, 1), dtype=np.float32), dtype=tf.float32, trainable=True)
l = tf.Variable(np.zeros_like(1.0), dtype=tf.float32, trainable=True)

w1 = tf.Variable(nn['w1'], trainable=False)
b1 = tf.Variable(nn['b1'], trainable=False)

w2 = tf.Variable(nn['w2'], trainable=False)
b2 = tf.Variable(nn['b2'], trainable=False)

w3 = tf.Variable(nn['w3'], trainable=False)
b3 = tf.Variable(nn['b3'], trainable=False)

w4 = tf.Variable(nn['w4'], trainable=False)
b4 = tf.Variable(nn['b4'], trainable=False)


def model():
    h1 = tf.nn.relu(tf.matmul(w1, i) + b1)
    h2 = tf.nn.relu(tf.matmul(w2, h1) + b2)
    h3 = tf.nn.relu(tf.matmul(w3, h2) + b3)
    h4 = tf.nn.sigmoid(tf.matmul(w4, h3) + b4)
    return h4


def evaluate_network(x):
    x_reshaped = np.array(x).reshape(2, 1).astype(np.float32)
    i.assign(x_reshaped)
    h4 = model()
    return h4[0][0].numpy()


def safety_predicate(val, label):
    return (val >= 0.5 and label >= 0.5) or (val < 0.5 and label < 0.5)


def fgsm(x, y, epsilon):
    x_reshaped = np.array(x).reshape(2, 1).astype(np.float32)
    y_reshaped = np.array(y).astype(np.float32)
    with tf.GradientTape() as tape:
        i.assign(x_reshaped)
        l.assign(y_reshaped)
        h4 = model()
        loss = tf.keras.losses.mean_squared_error(h4, l)
    input_grad = tape.gradient(loss, i)
    grad_sign = tf.math.sign(input_grad)
    scaled_grad = grad_sign * epsilon
    x_adv = x_reshaped + scaled_grad
    out = evaluate_network(x_adv)
    return x_adv, evaluate_network(x_adv), safety_predicate(out, y)

def repeated_fgsm(x, y, epsilon, r, segments=100):
    num_adv = 0
    num_total = 0
    for x0 in np.linspace(x[0] - r, x[0] + r, segments):
        for x1 in np.linspace(x[1] - r, x[1] + r, segments):
            x_adv, val, is_safe = fgsm([x0,x1], y, epsilon)
            num_total += 1
            if not is_safe:
                num_adv += 1
    return num_adv / num_total
