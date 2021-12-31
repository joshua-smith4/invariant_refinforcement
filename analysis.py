import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results_file', default='out.pickle')
parser.add_argument('--title', default='Invariant Reinforcement')

args = parser.parse_args()

with open(args.results_file, 'rb') as f:
    results = pickle.load(f)

var_class0 = results['var0']
var_class1 = results['var1']
mean_class0 = results['mean0']
mean_class1 = results['mean1']
num_correct = results['correct']
accuracy = results['acc']
epoch_acc = results['epoch_acc']

# class 0 variance
class0_fig = plt.figure(constrained_layout=True, figsize=(20, 8))
var_subfig_0, mean_subfig_0 = class0_fig.subfigures(2, 1)

var_subplts = var_subfig_0.subplots(1, 4)
var_subplts[0].plot(range(len(var_class0[0])), var_class0[0])
var_subplts[0].set_title('Layer 0')
var_subplts[1].plot(range(len(var_class0[1])), var_class0[1])
var_subplts[1].set_title('Layer 1')
var_subplts[2].plot(range(len(var_class0[2])), var_class0[2])
var_subplts[2].set_title('Layer 2')
var_subplts[3].plot(range(len(var_class0[3])), var_class0[3])
var_subplts[3].set_title('Layer 3')
var_subfig_0.suptitle('Variance', fontsize='x-large')


mean_subplts = mean_subfig_0.subplots(1, 4)
mean_subplts[0].plot(range(len(mean_class0[0])), mean_class0[0])
mean_subplts[1].plot(range(len(mean_class0[1])), mean_class0[1])
mean_subplts[2].plot(range(len(mean_class0[2])), mean_class0[2])
mean_subplts[3].plot(range(len(mean_class0[3])), mean_class0[3])
mean_subfig_0.suptitle('Mean', fontsize='x-large')

class0_fig.suptitle(f'Class 0 - {args.title}', fontsize='xx-large')

class1_fig = plt.figure(constrained_layout=True, figsize=(20, 8))
var_subfig_1, mean_subfig_1 = class1_fig.subfigures(2, 1)
var_class1_subplts = var_subfig_1.subplots(1, 4)
var_class1_subplts[0].plot(range(len(var_class1[0])), var_class1[0])
var_class1_subplts[0].set_title('Layer 0')
var_class1_subplts[1].plot(range(len(var_class1[1])), var_class1[1])
var_class1_subplts[1].set_title('Layer 1')
var_class1_subplts[2].plot(range(len(var_class1[2])), var_class1[2])
var_class1_subplts[2].set_title('Layer 2')
var_class1_subplts[3].plot(range(len(var_class1[3])), var_class1[3])
var_class1_subplts[3].set_title('Layer 3')
var_subfig_1.suptitle('Variance', fontsize='x-large')

mean_class1_subplts = mean_subfig_1.subplots(1, 4)
mean_class1_subplts[0].plot(range(len(mean_class1[0])), mean_class1[0])
mean_class1_subplts[1].plot(range(len(mean_class1[1])), mean_class1[1])
mean_class1_subplts[2].plot(range(len(mean_class1[2])), mean_class1[2])
mean_class1_subplts[3].plot(range(len(mean_class1[3])), mean_class1[3])
mean_subfig_1.suptitle('Mean', fontsize='x-large')

class1_fig.suptitle(f'Class 1 - {args.title}', fontsize='xx-large')

# batch accuracy
batch_accuracy = plt.figure()
acc_subplot = batch_accuracy.subplots(1,1)
acc_subplot.plot(range(len(accuracy)), accuracy)
batch_accuracy.suptitle(f'Accuracy - {args.title}', fontsize='xx-large')

plt.show()
