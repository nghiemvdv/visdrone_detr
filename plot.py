import argparse
# import sys
# import glob
# from util.plot_utils import plot_logs, plot_precision_recall, merge_logs
from pathlib import Path
import matplotlib.pyplot as plt


# path_obj = [Path(p) for p in glob.glob('/content/drive/My Drive/kltn/experiments/resnet50_2/10*_epoch')]
# merged = merge_logs(path_obj, '/content/drive/My Drive/kltn/plot')
# plot_logs(merged)
# paths = [Path(p) for p in glob.glob("/content/drive/My Drive/kltn/40_epoch_output/eval/*.pth")]
# plot_precision_recall(paths)E
def get_args_parser():
	parser = argparse.ArgumentParser('Plot', add_help=False)
	parser.add_argument('--plot_logs', default='', type=str)
	parser.add_argument('--plot_precision_recall', default='', type=str)
	return parser

def main(args):
	if args.plot_logs:
		plot_logs(Path(args.plot_logs))
	if args.plot_precision_recall:
		plot_precision_recall(args.plot_precision_recall)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('Plot script', parents=[get_args_parser()])
#     args = parser.parse_args()
#     main(args)
plot_logs(Path('/gdrive/kltn/experiments/detr-res101-dc5/50_epoch'))