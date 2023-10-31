import os
import sys
import numpy as np

def optimal_sci(n_s):
    if n_s <= 1:
        return np.nan
    return 1

def optimal_hit(n_s, k):
    return min(n_s, k)

avg_hit = dict.fromkeys([1,3,5,10], 0)
num_cells = 0

with open(sys.argv[1], 'r') as fp:
    next(fp)
    for line in fp:
        tmp = line.strip().split()
        cell = tmp[0]
        num_cells += 1
        n_s = int(tmp[2])

        to_print = cell
        for k in [1,3,5,10]:
            to_print += f',{optimal_hit(n_s, k)}'
            avg_hit[k] += optimal_hit(n_s, k)
        print(to_print)

print('Avg', end='')
for k in [1,3,5,10]:
    print(f',{avg_hit[k]/num_cells:.3f}', end='')
print()

