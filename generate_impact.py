#!/usr/bin/env python

# University of Pittsburgh
# Center for Simulation and Modeling
# Esteban Meneses
# Generates distributions from a failed job file 

import sys
import re
import datetime
import time
import matplotlib as mtl
mtl.use('Agg')
import matplotlib.pyplot as plt
from numpy import *

MAKE_PLOTS=1
COMPUTE_CORRELATION=0

### FUNCTIONS ###

def plot_histogram(data, bins, titleX, titleY, dirName, file_name):
	""" Saves into outputFile a histogram with data """
	outputFile = dirName + '/' + file_name + '.pdf'
	output_filename_txt = dirName + '/' + file_name + '.txt'
	plt.clf()
	plt.rc('font', family='DejaVu Sans')
	plt.rc('font', serif='Times New Roman')
	plt.rc('font', size=24)
	plt.yscale('log')
	hist_n, hist_bins, hist_patches = plt.hist(data, bins, histtype='bar', stacked=True, color=['Red','Blue'], label=['software','hardware'])
	tmp = plt.gcf().get_size_inches()
	plt.xlabel(titleX)
	plt.ylabel(titleY)
	plt.legend(fontsize='medium')
	plt.savefig(outputFile,bbox_inches='tight')
	output_file_txt = open(output_filename_txt, 'w')
	output_file_txt.write('#BIN VALUE\n')
	output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), hist_bins, hist_n)))
	output_file_txt.close()
	return

def plot_cum_histogram(data, bins, titleX, titleY, dirName, file_name):
	""" Saves into outputFile a histogram with data """
	outputFile = dirName + '/' + file_name + '.pdf'
	output_filename_txt = dirName + '/' + file_name + '.txt'
	plt.clf()
	plt.rc('font', family='DejaVu Sans')
	plt.rc('font', serif='Times New Roman')
	plt.rc('font', size=24)
	hist_n, hist_bins, hist_patches = plt.hist(data, bins, normed=1, cumulative=True)
	tmp = plt.gcf().get_size_inches()
	plt.xlabel(titleX)
	plt.ylabel(titleY)
	plt.ylim(0, 1.05)
	plt.xlim(0,18688)
	plt.savefig(outputFile,bbox_inches='tight')
	output_file_txt = open(output_filename_txt, 'w')	
	output_file_txt.write('#BIN VALUE\n')
	output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), hist_bins, hist_n)))
	output_file_txt.close()
	return

def plot_weight_histogram(data, bins, weights, titleX, titleY, dirName, file_name):
	""" Saves into outputFile a histogram with data """
	outputFile = dirName + '/' + file_name + '.pdf'
	output_filename_txt = dirName + '/' + file_name + '.txt'
	plt.clf()
	plt.rc('font', family='DejaVu Sans')
	plt.rc('font', serif='Times New Roman')
	plt.rc('font', size=24)
	hist_n, hist_bins, hist_patches = plt.hist(data, bins, weights=weights, normed=1, cumulative=True, stacked=True, color=['Red','Blue'],label=['software','hardware'])
	tmp = plt.gcf().get_size_inches()
	plt.xlabel(titleX)
	plt.ylabel(titleY)
	plt.ylim(0, 1.05)
	plt.xlim(0,18688)
	plt.legend(loc=2,fontsize='medium')
	plt.savefig(outputFile,bbox_inches='tight')
	output_file_txt = open(output_filename_txt, 'w')
	output_file_txt.write('#BIN VALUE\n')
	output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), hist_bins, hist_n)))
	output_file_txt.close()
	return

def plotScatter(dataX, dataY, titleX, titleY, dirName, file_name):
	""" Saves into outputFile a scatter plot with data """
	outputFile = dirName + '/' + file_name + '.pdf'
	plt.clf()
	plt.rc('font', family='DejaVu Sans')
	plt.rc('font', serif='Times New Roman')
	plt.rc('font', size=24)
	plt.yscale('log')
	plt.ylim([1,max(dataY)])
	plt.xlim([0,max(dataX)])
	plt.scatter(dataX, dataY)
	plt.xlabel(titleX)
	plt.ylabel(titleY)
	plt.savefig(outputFile,bbox_inches='tight')
	return

def distributions(fileName, dirName, bins):
	""" Reads a failed-job file and generates distributions in the directory """
	timeFormat = '%m/%d/%y %H:%M %p'
	formatAlt = '%Y-%m-%d %H:%M:%S'
	dayFormat = '%a_%b_%d_%Y'
	count = 0
	nodes_sw = []
	nodes_hw = []
	tasks_sw = []
	tasks_hw = []
	wallclock_sw = []
	wallclock_hw = []
	wait_sw = []
	wait_hw = []
	sus_sw = []
	sus_hw = []
	req_sus_sw = []
	req_sus_hw = []
	execution_sw = []
	execution_hw = []
	total_sus = 0
	total_jobs = 0

	# start timer
	startTime = time.clock()
	
	with open(fileName) as f:
		lines = len(f.readlines())
	lines = lines - 1

	# going through all entries in the file
	with open(fileName) as f:
		next(f)																			# skipping first line (header)
		for line in f:

			# getting jobid and time for each failure
			count = count + 1
			fields = line.split('\t')													# separating fields
			jobid = fields[0].strip()													# reading job ib
			dateAndTime = fields[1].strip()												# reading time
			req_nodes = int(fields[2])
			wallclock_time = int(fields[4])
			execution_time = float(fields[6])
			failure_type = fields[7].strip()
			total_jobs = total_jobs + 1
			total_sus = total_sus + (req_nodes * execution_time/60.0)/1000.0			# transforming into node-hours 
	
			if(failure_type == 'software'):
				nodes_sw.append(int(fields[2]))
				tasks_sw.append(int(fields[3]))
				wallclock_sw.append(int(fields[4]))
				wait_sw.append(float(fields[5]))
				execution_sw.append(execution_time)
				sus_sw.append(req_nodes*execution_time/60.0/1000.0)
				req_sus_sw.append(req_nodes*wallclock_time/60.0/1000.0)
			else:
				nodes_hw.append(int(fields[2]))
				tasks_hw.append(int(fields[3]))
				wallclock_hw.append(int(fields[4]))
				wait_hw.append(float(fields[5]))
				execution_hw.append(execution_time)
				sus_hw.append(req_nodes*execution_time/60.0/1000.0)
				req_sus_hw.append(req_nodes*wallclock_time/60.0/1000.0)
			
			print ("Progressing failed job file: %d%%" % (count/lines*100),end="\r") 
			sys.stdout.flush()

	# creating plots
	if(MAKE_PLOTS):
		print ("\nGenerating graphics 1 of 6") 
		plot_histogram([nodes_sw,nodes_hw], bins, 'Requested Nodes', 'Number of Jobs', dirName, 'impact_nodes')	
#		plot_cum_histogram(nodes, bins, 'Requested Nodes', 'Cumulative Fraction of Total Number of Jobs', dirName, 'impact_cum_nodes')	
		print ("Generating graphics 2 of 6") 
		plot_weight_histogram([nodes_sw,nodes_hw], bins, [sus_sw,sus_hw], 'Requested Nodes', 'Cumulative Fraction of Node SUs', dirName, 'impact_cum_sus_nodes')	
		print ("Generating graphics 3 of 6")
		plot_histogram([wallclock_sw,wallclock_hw], bins, 'Requested Wallclock Time (minutes)', 'Number of Jobs', dirName, 'impact_wallclock')	
		print ("Generating graphics 4 of 6")
		plot_histogram([execution_sw,execution_hw], bins, 'Execution Time (minutes)', 'Number of Jobs', dirName, 'impact_execution')
		wait_bins = range(0,20000,int(20000/bins))	
		print ("Generating graphics 5 of 6")
		plot_histogram([wait_sw,wait_hw], wait_bins, 'Wait Time (minutes)', 'Number of Jobs', dirName, 'impact_wait_time')	
		print ("Generating graphics 6 of 6\n")
		plot_histogram([sus_sw,sus_hw], bins, 'Node Service Units (thousands)', 'Number of Jobs', dirName, 'impact_sus')	

	# computing correlation coefficient between nodes and wallclock
	if(COMPUTE_CORRELATION):
		nodes_array = array(nodes)
		wallclock_array = array(wallclock)
		coefficient = corrcoef(nodes_array,wallclock_array)[1,0]	
		print ("Correlation coefficient (nodes vs wallclock): %f" % coefficient)

	# computing totals
	print ("Total software impact on nodes: %d" % sum(nodes_sw))
	print ("Total hardware impact on nodes: %d" % sum(nodes_hw))
	print ("Total software impact on execution: %f" % sum(execution_sw))
	print ("Total hardware impact on execution: %f" % sum(execution_hw))
	print ("Total software impact on wait time: %f" % sum(wait_sw))
	print ("Total hardware impact on wait time: %f" % sum(wait_hw))
	print ("Total software impact on sus: %f" % sum(sus_sw))
	print ("Total hardware impact on sus: %f" % sum(sus_hw))

	# stop timer
	finishTime = time.clock()

	print ("%d jobs analyzed in %.3f seconds" % (count, finishTime-startTime))
	
	return

### MAIN CODE ###
if len(sys.argv) >= 4:
	fileName = sys.argv[1]
	dirName = sys.argv[2]
	bins = int(sys.argv[3])
	distributions(fileName, dirName, bins)
else:
	print ("ERROR, usage: %s <file> <directory> <bins>" % sys.argv[0])
	print ("<file>: failed-job file")
	print ("<directory>: output directory")
	print ("<bins>: number of bins for histograms")
	sys.exit(0)

