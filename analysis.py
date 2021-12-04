import numpy as np

invariants_h1 = np.load('invariants_h1.npy')
invariants_h2 = np.load('invariants_h2.npy')
invariants_h3 = np.load('invariants_h3.npy')

all_invariants = np.concatenate((invariants_h1, invariants_h2, invariants_h3), axis=3)

def analyze_invariants(inv):
    fshared_neurons = np.zeros_like(inv[0][0]).astype(np.int32)
    lshared_neurons = np.zeros_like(inv[0][0]).astype(np.int32)
    inv_size = inv.shape[3]
    for i in range(10):
        print('class {}'.format(i))
        finv = inv[0][i] != -1
        linv = inv[3][i] != -1
        fshared_neurons = finv.astype(np.int32) + fshared_neurons
        lshared_neurons = linv.astype(np.int32) + lshared_neurons

        finv_low = inv[0][i] == 1
        finv_high = inv[0][i] == 2

        linv_low = inv[3][i] == 1
        linv_high = inv[3][i] == 2

        comb = np.logical_and(finv, linv)
        comb_dir = np.logical_or(np.logical_and(finv_low, linv_low), np.logical_and(finv_high, linv_high))
        finv_size = np.sum(finv)
        linv_size = np.sum(linv)
        comb_size = np.sum(comb)
        comb_dir_size = np.sum(comb_dir)
        print('size of first invariant: {}'.format(finv_size))
        print('percentage of total neurons: {}'.format(finv_size/inv_size*100))
        print('size of last invariant: {}'.format(linv_size))
        print('percentage of total neurons: {}'.format(linv_size/inv_size*100))
        print('size of conjunction between first and last: {}'.format(comb_size))
        print('percentage of size of last invariant: {}'.format(comb_size/linv_size*100))
        print('size of directional conjuction: {}'.format(comb_dir_size))
        print('directional conjunction ratio: {}'.format(comb_dir_size/comb_size * 100))
        print()
    print('shared first: \n{}'.format(fshared_neurons))
    print('shared last: \n{}'.format(lshared_neurons))

print('-----------------------invariants h1')
analyze_invariants(invariants_h1)
print()

print('-----------------------invariants h2')
analyze_invariants(invariants_h2)
print()

print('-----------------------invariants h3')
analyze_invariants(invariants_h3)
print()

print("-----------------------all invariants")
analyze_invariants(all_invariants)
print()
