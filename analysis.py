import numpy as np
import pickle
import matplotlib.pyplot as plt

with open('results/sgd/out.pickle', 'rb') as f:
    results = pickle.load(f)

var_class0 = results['var0']
var_class1 = results['var1']
mean_class0 = results['mean0']
mean_class1 = results['mean1']
num_correct = results['correct']
accuracy = results['acc']
epoch_acc = results['epoch_acc']

# class 0 variance
plt.figure()
plt.plot(range(len(var_class0[0])), var_class0[0])
plt.title('variance class 0 neuron 0 layer 0')
plt.figure()
plt.plot(range(len(var_class0[1])), var_class0[1])
plt.title('variance class 0 neuron 0 layer 1')
plt.figure()
plt.plot(range(len(var_class0[2])), var_class0[2])
plt.title('variance class 0 neuron 0 layer 2')
plt.figure()
plt.plot(range(len(var_class0[3])), var_class0[3])
plt.title('variance class 0 neuron 0 layer 3')

# class 1 variance
plt.figure()
plt.plot(range(len(var_class1[0])), var_class1[0])
plt.title('variance class 1 neuron 0 layer 0')
plt.figure()
plt.plot(range(len(var_class1[1])), var_class1[1])
plt.title('variance class 1 neuron 0 layer 1')
plt.figure()
plt.plot(range(len(var_class1[2])), var_class1[2])
plt.title('variance class 1 neuron 0 layer 2')
plt.figure()
plt.plot(range(len(var_class1[3])), var_class1[3])
plt.title('variance class 1 neuron 0 layer 3')

# batch accuracy
plt.figure()
plt.plot(range(len(accuracy)), accuracy)
plt.title('batch accuracy')

# class 0 mean
plt.figure()
plt.plot(range(len(mean_class0[0])), mean_class0[0])
plt.title('mean class 0 neuron 0 layer 0')
plt.figure()
plt.plot(range(len(mean_class0[1])), mean_class0[1])
plt.title('mean class 0 neuron 0 layer 1')
plt.figure()
plt.plot(range(len(mean_class0[2])), mean_class0[2])
plt.title('mean class 0 neuron 0 layer 2')
plt.figure()
plt.plot(range(len(mean_class0[3])), mean_class0[3])
plt.title('mean class 0 neuron 0 layer 3')

# class 1 mean
plt.figure()
plt.plot(range(len(mean_class1[0])), mean_class1[0])
plt.title('mean class 1 neuron 0 layer 0')
plt.figure()
plt.plot(range(len(mean_class1[1])), mean_class1[1])
plt.title('mean class 1 neuron 0 layer 1')
plt.figure()
plt.plot(range(len(mean_class1[2])), mean_class1[2])
plt.title('mean class 1 neuron 0 layer 2')
plt.figure()
plt.plot(range(len(mean_class1[3])), mean_class1[3])
plt.title('mean class 1 neuron 0 layer 3')

plt.show()
