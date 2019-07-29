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

	args = parser.parse_args()

	args.output_folder = os.path.join(args.output_dir, args.output_folder+'{:.2f}')

	steps = np.arange(0.1, 0.8, 0.05)
	# steps = [0.1, 0.2, 0.3]
	# steps = [0.1]

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)


	if not args.visual_only:
		if args.muti:
			p = Pool(4)
			cmds = list(map(mkcmd, steps, repeat(args.output_folder)))
			result = p.map(trainMain, cmds)
		else:
			result = []
			for step in steps:
				result.append(trainMain(mkcmd(step, args.output_folder)))

		names = ["time_step", "success_rate", "collision_rate", "avg_nav_time", "total_reward", "freq_danger", "ave_min_dist"]

		with open(os.path.join(args.output_dir, args.vis_txt_dir), 'w+') as f:
			f.write(", ".join(names) + "\n")
			for i, lst in enumerate(result):
				f.write("{:.2}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n".format(lst[6], lst[0], lst[1], lst[2], lst[3], lst[4], lst[5]))

			f.close()

	if not os.path.exists(os.path.join(args.output_dir, args.vis_txt_dir)):
		print("Could not find {} file.".format(args.vis_txt_dir))
		print("Exiting...")
		sys.exit(1)

	if not os.path.exists(os.path.join(args.output_dir, args.figure_dir)):
		os.makedirs(os.path.join(args.output_dir, args.figure_dir))

	visualize(args.output_dir, args.figure_dir)



