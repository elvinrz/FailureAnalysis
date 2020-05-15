
#!/usr/bin/env python

# University of Pittsburgh
# Center for Simulation and Modeling
# Esteban Meneses
# Date: 03/20/15


import matplotlib.pyplot as plt
import numpy as np
import sys
import math

	
def plot_sockets(file_name, diroutput):
	line_count = 0
	x = []
	y = []
	x_gpu = []
	y_gpu = []
	x_mem = []
	y_mem = []
	
	
	#for plot sockets
	file_name1 = file_name + "sockets.dat"
	with open(file_name1) as log:
		line_count += 1
		if line_count == 1:
			next(log)
		for line in log:
			item = line.split()
			x.append(int(item[0]))
			y.append(float(item[1]))
	
	#for plot GPU

	file_name2 = file_name + "accps.dat"
	with open(file_name2) as log:
		line_count += 1
		if line_count == 1:
			next(log)
		for line in log:
			item = line.split()
			x_gpu.append(int(item[0]))
			y_gpu.append(float(item[1]))

	print(y_gpu)
	
	file_name3 = file_name + "mem.dat"
	with open(file_name3) as log:
		line_count += 1
		if line_count == 1:
			next(log)
		for line in log:
			item = line.split()
			x_mem.append(int(item[0]))
			y_mem.append(float(item[1]))	

	print("\nProcessing plot ...")	
	
	

	plt.clf()
	fig, axs = plt.subplots(1,3,figsize=(10, 2.3))
	ax = plt.gca()
	ind = np.arange(2005,2020,1)

	data_line = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]

	labels = ["2005","","2007","","2009", "","2010","","2012", "","2013","","2015", "","2016","","2017", "","2018","","2019"]
	
	axs[1].scatter(x, y, s=10, c="blue", marker='s', edgecolor='black', linewidth='0.1', alpha=0.3,)
	axs[2].scatter(x_gpu, y_gpu, s=10, c="red",marker='s', edgecolor='black', linewidth='0.2', alpha=0.3)
	axs[0].scatter(x_mem, y_mem, s=10, c="green",marker='s', edgecolor='black', linewidth='0.2', alpha=0.3)
		
	#axs2 = axs[1].twinx()
	#axs2.plot(ind,data_line,'C1', linewidth=0.5,label='MTTI')
	#axs2.tick_params(axis='y',labelsize = 7)
	#axs2.set_ylabel('Machine MTTI (minutes)',fontsize=8)
	
	axs[1].set_xticks(np.arange(2005,2020,1 ), labels)
	axs[1].tick_params(axis='x',labelsize = 7.5)
	axs[1].tick_params(axis='y',labelsize = 7)
	#axs[1].set_yscale("log")
	#axs[1].legend(edgecolor="black",prop={'size': 6})	
	axs[1].set_xlabel('Year',fontsize=11)
	axs[1].set_ylabel('Number of Sockets',fontsize=11)
	
	axs[2].set_xticks(np.arange(2005,2020,1 ), labels)
	axs[2].tick_params(axis='x',labelsize = 7.5)
	axs[2].tick_params(axis='y',labelsize = 7)
	#axs[2].set_yscale("log")
	#axs[2].legend(edgecolor="black",prop={'size': 6})	
	axs[2].set_xlabel('Year',fontsize=11)
	axs[2].set_ylabel('ACC per Socket',fontsize=11)
	
	axs[0].set_xticks(np.arange(2005,2020,1 ), labels)
	axs[0].tick_params(axis='x',labelsize = 7.5)
	axs[0].tick_params(axis='y',labelsize = 7)
	#axs[0].set_yscale("log")
	#axs[0].legend(edgecolor="black",prop={'size': 6})	
	axs[0].set_xlabel('Year',fontsize=11)
	axs[0].set_ylabel('Memory per Socket (GB)',fontsize=11)
	
	plt.tight_layout()
	plt.savefig(diroutput+"socket.pdf")
	print("\nPlot in file: <socket.pdf>")


	return 
	
	
if len(sys.argv) >= 3:
	dirName = sys.argv[1]
	diroutput = sys.argv[2]
	plot_sockets(dirName,diroutput)
else:
	print ("ERROR, usage: %s <directory> <output dir>" % sys.argv[0])
	sys.exit(0)
