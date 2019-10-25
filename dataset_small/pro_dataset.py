#!/usr/bin/env python3
import os
dir_name = "./train/"
dir_name2 = "./test/"

for i in range(15):

	labels = sorted([int(x) for x in os.listdir(dir_name)])
	labels_str = sorted([x for x in os.listdir(dir_name)])

	wrong_id = []

	for idx in range(len(labels)-1):
		if (labels[idx+1] - labels[idx]) > 1:
				wrong_id.append(idx)
	print(wrong_id)
	input()
	count = 0
	wrong_id_str = ["{:0>5d}".format(value) for value in wrong_id]
	# for idx in wrong_id:
	idx = wrong_id[0]
	print(labels_str[idx])
	for mv_id in range(idx+1, len(labels_str)-1):
		source = labels_str[mv_id-count]
		target = "{:0>5d}".format(int(source)-1)
		cmd = "mv %s %s" % (dir_name+source, dir_name+target)
		cmd2 = "mv %s %s" % (dir_name2+source, dir_name2+target)

		print(cmd)
		print(cmd2)
		os.system(cmd)
		os.system(cmd2)
