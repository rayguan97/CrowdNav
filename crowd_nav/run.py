#!/anaconda2/envs/gamma/bin/python

import os
import sys
import argparse
import numpy as np
from multiprocessing import Pool
from itertools import repeat

from train import main as trainMain
from test import main as testMain
from visual import visualize

OUTPATH = "data/output"

def mkcmd(step, out_folder):
	return ['--output_dir', out_folder.format(step), '--time_step', str(step)]




if __name__ == '__main__':


	parser = argparse.ArgumentParser('Parse visualization')
	parser.add_argument('--output_dir', default=OUTPATH, type=str)
	parser.add_argument('--output_folder', default='step', type=str)
	parser.add_argument('--figure_dir', default='figure', type=str)
	parser.add_argument('--vis_txt_dir', default='visualization.txt', type=str)
	parser.add_argument('--muti', default=False, action='store_true')
	parser.add_argument('--visual_only', '-v', default=False, action='store_true')
	parser.add_argument('--append', '-a', default=False, action='store_true')
	args = parser.parse_args()

	args.output_folder = os.path.join(args.output_dir, args.output_folder+'{:.2f}')
	args.mode = 'a+' if args.append else 'w+'

	steps = np.arange(0.1, 1, 0.05)
	# steps = [0.1, 0.2, 0.3]
	# steps = [0.1]

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)


	if not args.visual_only:

		names = ["Time_step", "Success_rate", "Collision_rate", "Average_navigation_time", "Total_reward", "Frequency_of_in_danger", "Average_min_seperating_dist_in_danger", "Average_loss", "Training_time (seconds)"]


		if args.muti:
			p = Pool(4)
			cmds = list(map(mkcmd, steps, repeat(args.output_folder)))
			result = p.map(trainMain, cmds)

			with open(os.path.join(args.output_dir, args.vis_txt_dir), args.mode) as f:
				if not args.append:
					f.write(", ".join(names) + "\n")
				for lst in result:
					f.write("{:.2}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.2f}\n".format(lst[8], lst[0], lst[1], lst[2], lst[3], lst[4], lst[5], lst[6], lst[7]))

				f.close()

		else:
			result = []

			with open(os.path.join(args.output_dir, args.vis_txt_dir), args.mode) as f:
				if not args.append:
					print("Here")
					print(args.mode)
					print(", ".join(names) + "\n")
					f.write(", ".join(names) + "\n")
					f.flush()

				for step in steps:
					print('Running time_step = {}'.format(step))
					lst = trainMain(mkcmd(step, args.output_folder))
					result.append(lst)
					f.write("{:.2}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.2f}\n".format(lst[8], lst[0], lst[1], lst[2], lst[3], lst[4], lst[5], lst[6], lst[7]))
					f.flush()
					
				f.close()


	if not os.path.exists(os.path.join(args.output_dir, args.vis_txt_dir)):
		print("Could not find {} file.".format(args.vis_txt_dir))
		print("Exiting...")
		sys.exit(1)

	if not os.path.exists(os.path.join(args.output_dir, args.figure_dir)):
		os.makedirs(os.path.join(args.output_dir, args.figure_dir))

	visualize(args.output_dir, args.figure_dir)



