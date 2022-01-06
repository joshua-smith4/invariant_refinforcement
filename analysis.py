import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--results_files', nargs='+', dest='results_files')
parser.add_argument('--dataset_names', nargs='+')

args = parser.parse_args()

if args.results_files is None or len(args.results_files) <= 0:
    parser.print_help()
    exit()
if args.dataset_names is None or len(args.dataset_names) <= 0:
    args.dataset_names = [ x.split('/')[-1].split('.')[0] for x in args.results_files]

observed_neurons = [(0, 0), (1, 0), (2, 0), (3, 0),
                    (0, 1), (0, 2), (0, 3), (0, 4),
                    (1, 1), (1, 2), (1, 3), (1, 4),
                    (2, 1), (2, 2), (2, 3), (2, 4)]

results = []
for res_file in args.results_files:
    with open(res_file, 'rb') as f:
        results.append(pickle.load(f))

variance_fig_0 = plt.figure(constrained_layout=True, figsize=(20, 8))
variance_fig_1 = plt.figure(constrained_layout=True, figsize=(20, 8))
variance_fig_0.suptitle('Variance Class 0')
variance_fig_1.suptitle('Variance Class 1')
variance_subplots_0 = variance_fig_0.subplots(4, 4)
variance_subplots_1 = variance_fig_1.subplots(4, 4)

mean_fig_0 = plt.figure(constrained_layout=True, figsize=(20, 8))
mean_fig_1 = plt.figure(constrained_layout=True, figsize=(20, 8))
mean_fig_0.suptitle('Mean Class 0')
mean_fig_1.suptitle('Mean Class 1')
mean_subplots_0 = mean_fig_0.subplots(4, 4)
mean_subplots_1 = mean_fig_1.subplots(4, 4)

accuracy_fig = plt.figure(constrained_layout=True, figsize=(20, 8))
accuracy_fig.suptitle('Accuracy')
accuracy_plt = accuracy_fig.subplots(1, 1)

for index, result in enumerate(results):
    var_class0 = result['var0']
    var_class1 = result['var1']
    mean_class0 = result['mean0']
    mean_class1 = result['mean1']
    num_correct = result['correct']
    accuracy = result['acc']
    epoch_acc = result['epoch_acc']
    for i in range(4):
        for j in range(4):
            flatIndex = i*4 + j
            subplt = variance_subplots_0[i, j]
            subplt.plot(var_class0[flatIndex], label=f'{args.dataset_names[index]}')
            _l, _i = observed_neurons[flatIndex]
            subplt.set_title(f'Layer {_l} Index {_i}')
            subplt.legend()
    for i in range(4):
        for j in range(4):
            flatIndex = i*4 + j
            subplt = variance_subplots_1[i, j]
            subplt.plot(var_class1[flatIndex], label=f'{args.dataset_names[index]}')
            _l, _i = observed_neurons[flatIndex]
            subplt.set_title(f'Layer {_l} Index {_i}')
            subplt.legend()
    for i in range(4):
        for j in range(4):
            flatIndex = i*4 + j
            subplt = mean_subplots_0[i, j]
            subplt.plot(mean_class0[flatIndex], label=f'{args.dataset_names[index]}')
            _l, _i = observed_neurons[flatIndex]
            subplt.set_title(f'Layer {_l} Index {_i}')
            subplt.legend()
    for i in range(4):
        for j in range(4):
            flatIndex = i*4 + j
            subplt = mean_subplots_1[i, j]
            subplt.plot(mean_class1[flatIndex], label=f'{args.dataset_names[index]}')
            _l, _i = observed_neurons[flatIndex]
            subplt.set_title(f'Layer {_l} Index {_i}')
            subplt.legend()
    accuracy_plt.plot(accuracy, label=f'{args.dataset_names[index]}')
    accuracy_plt.legend()

plt.show()
