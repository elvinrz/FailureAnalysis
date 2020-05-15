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
FILTER_SMALL=0
SMALL_LIMIT=126

### CLASSES ###

class Job:
	""" This class represents a failed job and its attributes """
	def __init__(self, failureDate, nodes, tasks, wallclockTime, waitTime, execution_time):
		self.failureDate = failureDate
		self.nodes = nodes
		self.tasks = tasks
		self.wallclockTime = wallclockTime
		self.waitTime = waitTime
		self.execution_time = execution_time

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
	hist_n, hist_bins, hist_patches = plt.hist(data, bins)
	plt.xlabel(titleX)
	plt.ylabel(titleY)
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
	hist_n, hist_bins, hist_patches = plt.hist(data, bins, weights=weights, normed=1, cumulative=True)
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
	nodes = []
	tasks = []
	wallclock = []
	wait = []
	sus = []
	req_sus = []
	execution = []
	total_sus = 0
	total_jobs = 0
	small_jobs = 0
	small_sus = 0

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
			total_jobs = total_jobs + 1
			total_sus = total_sus + (req_nodes * execution_time/60.0)/1000.0			# transforming into node-hours 
			if(FILTER_SMALL and req_nodes < SMALL_LIMIT):
				small_jobs = small_jobs + 1
				small_sus = small_sus + (req_nodes * execution_time/60.0)/1000.0
				continue
			nodes.append(int(fields[2]))
			tasks.append(int(fields[3]))
			wallclock.append(int(fields[4]))
			wait.append(float(fields[5]))
			execution.append(execution_time)
			sus.append(req_nodes*execution_time/60.0/1000.0)
			req_sus.append(req_nodes*wallclock_time/60.0/1000.0)
			
			print ("Progressing failed job file: %d%%" % (count/lines*100),end="\r") 
			sys.stdout.flush()

	#		if(req_nodes < 313 and execution_time > 750):
	#			print "JOB EXECUTION TIME OVER LIMIT: %s\n" % jobid
	
	# creating plots
	if(MAKE_PLOTS):
		print ("\nGenerating graphics 1 of 15")
		plot_histogram(nodes, bins, 'Requested Nodes', 'Number of Jobs', dirName, 'workload_nodes')	
		print ("\nGenerating graphics 2 of 15")
		plot_cum_histogram(nodes, bins, 'Requested Nodes', 'Cumulative Fraction of Total Number of Jobs', dirName, 'workload_cum_nodes')	
		print ("\nGenerating graphics 3 of 15")
		plot_weight_histogram(nodes, bins, sus, 'Requested Nodes', 'Cumulative Fraction of Total Node Service Units', dirName, 'workload_cum_sus_nodes')	
		print ("\nGenerating graphics 4 of 15")
		plot_histogram(wallclock, bins, 'Requested Wallclock Time (minutes)', 'Number of Jobs', dirName, 'workload_wallclock')	
		print ("\nGenerating graphics 5 of 15")
		plot_histogram(execution, bins, 'Execution Time (minutes)', 'Number of Jobs', dirName, 'workload_execution')	
		print ("\nGenerating graphics 6 of 15")
		plot_histogram(wait, bins, 'Wait Time (minutes)', 'Number of Jobs', dirName, 'workload_wait_time')	
		print ("\nGenerating graphics 7 of 15")
		plot_histogram(sus, bins, 'Node Service Units (thousands)', 'Number of Jobs', dirName, 'workload_sus')	
		print ("\nGenerating graphics 8 of 15")
		plotScatter(nodes, wait, 'Requested Nodes', 'Wait Time (minutes)', dirName, 'workload_nodes_wait_time')
		print ("\nGenerating graphics 9 of 15")
		plotScatter(wallclock, wait, 'Requested Wallclock Time (minutes)', 'Wait Time (minutes)', dirName, 'workload_wallclock_wait_time')
		print ("\nGenerating graphics 10 of 15")
		plotScatter(execution, wait, 'Execution Time (minutes)', 'Wait Time (minutes)', dirName, 'workload_execution_wait_time')
		print ("\nGenerating graphics 11 of 15")
		plotScatter(req_sus, wait, 'Requested Node Service Units (thousands)', 'Wait Time (minutes)', dirName, 'workload_req_sus_wait_time')
		print ("\nGenerating graphics 12 of 15")
		plotScatter(nodes, wallclock, 'Requested Nodes', 'Requested Wallclock Time (minutes)', dirName, 'workload_nodes_wallclock_time')
		print ("\nGenerating graphics 13 of 15")
		plotScatter(nodes, execution, 'Requested Nodes', 'Execution Time (minutes)', dirName, 'workload_nodes_execution_time')
		titan_bins = [1,126,313,3750,11250,18688]
		print ("\nGenerating graphics 14 of 15")
		plot_histogram(nodes, titan_bins, 'Requested Nodes', 'Number of Jobs', dirName, 'workload_nodes_titan')	
		print ("\nGenerating graphics 15 of 15")
		plot_weight_histogram(nodes, titan_bins, sus, 'Requested Nodes', 'Cumulative Fraction of Total Node Service Units', dirName, 'workload_cum_sus_nodes_titan')	

	# computing totals
	print ("Total jobs: %d" % total_jobs)
	print ("Total sus: %f" % total_sus)	

	# computing correlation coefficient between nodes and wallclock
	if(COMPUTE_CORRELATION):
		nodes_array = array(nodes)
		wallclock_array = array(wallclock)
		wait_array = array(wait)
		coefficient = corrcoef(nodes_array,wallclock_array)[1,0]	
		print ("Correlation coefficient (nodes vs wallclock): %f" % coefficient)
		coefficient = corrcoef(nodes_array,wait_array)[1,0]	
		print ("Correlation coefficient (nodes vs wait time): %f" % coefficient)

	# printing small job statistics
	if(FILTER_SMALL):
		print ("Fraction of small jobs: %f" % (small_jobs/float(total_jobs)))
		print ("Fraction of small sus: %f" % (small_sus/float(total_sus)))

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

