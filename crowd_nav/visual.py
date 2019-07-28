import numpy as np
import matplotlib.pyplot as plt
import os

def visualize(data_dir, figure_dir, file_name='visualization.txt'):


	data = {}
	# read data:
	with open(os.path.join(data_dir, file_name)) as f:

		line = f.readline()
		items = [ x.strip() for x in line.split(',') ]

		x_axis = items[0]

		for item in items:
			data[item] = []

		line = f.readline()

		while line:
			data_points = [ x.strip() for x in line.split(',') ]
			for i, item in enumerate(items):
				data[item].append(data_points[i])
			line = f.readline()


	for y_axis in items[1:]:
		title = x_axis + ' v.s. ' + y_axis

		fig = plt.figure(title)
		plt.plot(data[x_axis], data[y_axis], linestyle='-',label='')
		plt.xlabel(x_axis)
		plt.ylabel(y_axis)
		plt.title(title)
		plt.legend()
		fig.show()
		fig.savefig(os.path.join(data_dir, figure_dir, title + '.png'))
		fig.clear()





if __name__ == '__main__':
	visualize('data/output', 'figure')


