import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

i = tf.keras.layers.InputLayer(input_shape=(28, 28, 1))
fl = tf.keras.layers.Flatten()
h1 = tf.keras.layers.Dense(32, activation='relu')
h2 = tf.keras.layers.Dense(32, activation='relu')
h3 = tf.keras.layers.Dense(32, activation='relu')
o = tf.keras.layers.Dense(10) 
m_h1 = tf.keras.Sequential([i,fl,h1])
m_h2 = tf.keras.Sequential([i,fl,h1,h2])
m_h3 = tf.keras.Sequential([i,fl,h1,h2,h3])
model = tf.keras.Sequential([i,fl,h1,h2,h3,o])

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

train_dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(train_x[..., tf.newaxis] / 255, tf.float32),
     tf.cast(train_y, tf.int32)))
dataset = train_dataset.shuffle(1000).batch(1)

test_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(test_x[..., tf.newaxis] / 255, tf.float32),
            tf.cast(test_y, tf.int32)))
test_dataset = test_dataset.batch(64)

optimizer = tf.keras.optimizers.Adam()
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_history = []
invariants = np.zeros((3,1,32), dtype=np.float32)
direction = 1
def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss_value = loss_obj(labels, logits)
    loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, model.trainable_variables)
    inv_index = 0
    for ind, grad in enumerate(grads):
        if grad.shape != (32,):
            continue
        grads[ind] += invariants[inv_index][0] / 10 * abs(np.average(grad)) * direction
        inv_index += 1
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

model.compile(optimizer, loss_obj, metrics=['accuracy'])

def pred_correctly_classified(x, y):
    cl = tf.argmax(model(x[tf.newaxis, ...]), axis=1)
    return tf.equal(cl[0], tf.cast(y, tf.int64))

def pred_filter(desired_label):
    def filt(x, y):
        return tf.equal(desired_label, y)
    return filt

invariant_recalc_interval = 100
test_acc_interval = 100
all_invariants = []
loss_and_acc_values = []

def train(epochs):
    global invariants
    for epoch in range(epochs):
        for (batch, (images, labels)) in enumerate(dataset):
            if batch % invariant_recalc_interval == 0:
                #pass
                invariants = np.sum(report_invariants(), axis=0)
            train_step(images, labels)
            if batch % test_acc_interval == 0:
                loss_and_acc_values.append(model.evaluate(test_dataset))
                print(loss_and_acc_values[-1])
        print('Epoch {} finished'.format(epoch))

def report_invariants():
    correctly_classified = train_dataset.filter(pred_correctly_classified)
    class_filters = []
    invariants = []
    inv = np.zeros(((3,)+m_h1(train_x[0:1]).numpy().shape), dtype=np.float32)
    for c in range(10):
        class_filters.append(correctly_classified.filter(pred_filter(c)))
        invariants.append(np.zeros_like(inv) + 2)
    for class_index, c in enumerate(class_filters):
        print('calculating invariants for class {}'.format(class_index))
        for image, label in c.as_numpy_iterator():
            layer_activations_h1 = m_h1(image[tf.newaxis,...]).numpy()
            layer_activations_h2 = m_h2(image[tf.newaxis,...]).numpy()
            layer_activations_h3 = m_h3(image[tf.newaxis,...]).numpy()
            layers = [layer_activations_h1, layer_activations_h2, layer_activations_h3]
            for h_index, h_layer in enumerate(layers):
                for relu_index in range(h_layer.shape[1]):
                    inv_element = invariants[class_index][h_index][0][relu_index]
                    layer_element = h_layer[0][relu_index]
                    if inv_element == 0:
                        continue
                    if inv_element > 1:
                        invariants[class_index][h_index][0][relu_index] = -1 if layer_element == 0 else 1
                        continue
                    if inv_element == 1 and layer_element == 0:
                        invariants[class_index][h_index][0][relu_index] = 0
                        continue
                    if inv_element == -1 and layer_element != 0:
                        invariants[class_index][h_index][0][relu_index] = 0
                        continue
        if invariants[class_index][0][0][0] > 1:
            invariants[class_index] = inv
    return np.array(invariants)

train(epochs=5)
np.save('loss_and_acc_with_inv.npy', np.array(loss_and_acc_values))
