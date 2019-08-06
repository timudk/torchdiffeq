import numpy as np  
import matplotlib.pyplot as plt
from random import shuffle
import pickle
import argparse
import os
import sys

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def peak_function(x, y):
	return 3*(1-x)**2*np.exp(-x**2-(y+1)**2)-10*(x/5-x**3-y**5)*np.exp(-x**2-y**2)-(1/3)*np.exp(-(x+1)**2-y**2)

def grid_points(length, n_points_per_direction):
	discretization_1d = np.linspace(-length/2, length/2, n_points_per_direction)
	xx, yy = np.meshgrid(discretization_1d, discretization_1d, sparse=False)
	return xx, yy

def visualize(xx, yy, zz):
	h = plt.contourf(xx, yy, zz)
	plt.show()

def make_continuous_dataset(xx, yy, zz):
	x = xx[0, :]
	y = yy[:, 0]

	dataset = []
	for i in range(len(x)):
		for j in range(len(y)):
			dataset.append((x[i], y[j], zz[j, i]))

	return dataset

def sort_continuous_dataset(unsorted_dataset):
	return sorted(unsorted_dataset, key=lambda points:points[2])

def classify_sorted_dataset(sorted_dataset, n_classes, n_points_per_direction, print_list):
	n_points = n_points_per_direction**2
	points_per_class = int(n_points/n_classes)

	if print_list:
		separation = []

	separated_classes = []
	for i in range(n_classes):
		new_class = []

		if print_list:
			separation.append(sorted_dataset[i*points_per_class][2])
			separation.append(sorted_dataset[(i + 1)*points_per_class - 1][2])

		for j in range(points_per_class):
			new_class.append((sorted_dataset[i*points_per_class+j][0], sorted_dataset[i*points_per_class+j][1], i))

		separated_classes.append(new_class)

	if print_list:
		return separated_classes, np.array(separation)
	else:
		return separated_classes


def main(LENGTH, N_POINTS_PER_DIR, N_CLASSES, SPLIT, PRINT_LIST, VISUALIZE):
	if not os.path.isdir('classes_' + str(N_CLASSES)):
		os.mkdir('classes_' + str(N_CLASSES))

	name = 'classes_' + str(N_CLASSES) + '/length_' + str(LENGTH) + '_points_' + str(N_POINTS_PER_DIR) + '.pkl'
	if os.path.exists(name):
		print('File already exists and should not be overwritten.')
		return -1

	if PRINT_LIST:
		separation_name = 'classes_' + str(N_CLASSES) + '/length_' + str(LENGTH) + '_points_' + str(N_POINTS_PER_DIR) + '.txt'

	xx, yy = grid_points(LENGTH, N_POINTS_PER_DIR)
	zz = peak_function(xx, yy)

	if VISUALIZE:
		visualize(xx, yy, zz)

	continous_dataset = make_continuous_dataset(xx, yy, zz)
	sorted_continuous_dataset = sort_continuous_dataset(continous_dataset)

	if PRINT_LIST:
		separated_classes, separation = classify_sorted_dataset(sorted_continuous_dataset, N_CLASSES, N_POINTS_PER_DIR, PRINT_LIST)
	else:
		separated_classes = classify_sorted_dataset(sorted_continuous_dataset, N_CLASSES, N_POINTS_PER_DIR, PRINT_LIST)
	
	flatted_list = [item for sublist in separated_classes for item in sublist]
	shuffle(flatted_list)

	training_list = flatted_list[:int(SPLIT*N_POINTS_PER_DIR**2)]
	test_list = flatted_list[int(SPLIT*N_POINTS_PER_DIR**2):]

	print('Number of training points:', len(training_list))
	print('Number of test points:', len(test_list))

	with open(name, 'wb') as f:
		pickle.dump([training_list, test_list], f)

	if PRINT_LIST:
		np.savetxt(separation_name, separation)
		

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--length', type=int, default=6)
	parser.add_argument('--npoints', type=int, default=80)
	parser.add_argument('--nclasses', type=int, default=5)
	parser.add_argument('--split', type=float, default=0.8)
	parser.add_argument('--printlist', type=str2bool, default=False)
	parser.add_argument('--visualize', type=str2bool, default=False)
	args = parser.parse_args()

	main(args.length, 
		 args.npoints, 
		 args.nclasses, 
		 args.split,
		 args.printlist,
		 args.visualize)