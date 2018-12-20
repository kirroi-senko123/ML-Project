import numpy as np



labels = np.load("num_to_symbol_map.npy")

print((labels[0].shape))

for key in labels.items():
	if  key == 'ascii_124':
		key = '1'		
	if  key == 'times':
		key = 'x'
	if  key == 'sum':
		key = '\sigma'
	if  key == 'lambda' or labels[i] == 'Delta' or  labels[i] == 'neq' or  labels[i] == 'geq' or labels[i] == 'leq' or labels[i] == 'infty' or labels[i] == 'beta':
		key = "\\" +	labels[i]	

print(labels)