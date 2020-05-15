#!/usr/bin/env python

# University of Pittsburgh
# Center for Simulation and Modeling
# Esteban Meneses
# Extract failures from the failure log using a list of job ids.

import sys
import re
import datetime
import time
import matplotlib.pyplot as plt
from numpy import *

### FUNCTIONS ###

def extract(failure_file_name, filter_file_name, output_file_name):
	""" Reads the failure log file and extracts only the failures in the filter file """

	# data structures
	job_id_set = {}
	count = 0
	count_lines = 0
	remaining_keys = 0
	
	# start timer
	startTime = time.clock()

	# number of lines
	#size of failure file
	with open(failure_file_name) as f:
		linesFailureFile = len(f.readlines())
	
	with open(filter_file_name) as f:
		linesFilterJobFile = len(f.readlines())
	
	# collecting the set of job ids
	with open(filter_file_name) as f:
		for line in f:
			count_lines = count_lines + 1
			fields = line.split()
			job_id = fields[0].strip()
			job_id_set[job_id] = 1
			print ("Progress Filter Jobs: %d%%"% (count_lines/linesFilterJobFile*100),end="\r") 
			sys.stdout.flush()
			
	count_lines = 0		
	# traversing the list of failures
	output_file = open(output_file_name, 'w')
	print("\n")
	output_file.write("hostname\tjob_id\tfail_time\tcategory\treason\tdescription\ttext\n")
	with open(failure_file_name) as f:
		next(f)																			# skipping first line (header)
		for line in f:
			count_lines = count_lines + 1
			fields = line.split('|')													# separating fields
			job_id = fields[2].strip()													# reading job ib
			if(job_id in job_id_set.keys()):
				count = count + 1
				job_id_set.pop(job_id,None)
				output_file.write(line)

			print ("Progress Filter Failulre: %d%%"% (count_lines/linesFailureFile*100),end="\r") 
			sys.stdout.flush()

	output_file.close()

	#checking for remaining job ids
	for key in job_id_set.keys():
		remaining_keys = remaining_keys + 1
		
	# stop timer
	finishTime = time.clock()
    
	print("\n")
	print ("%d Jobs withouth failure" % (remaining_keys))	
	print ("%d jobs analyzed in %.3f seconds" % (count, finishTime-startTime))	
	return

### MAIN CODE ###
if len(sys.argv) >= 4:
	failure_file_name = sys.argv[1]
	filter_file_name = sys.argv[2]
	output_file_name = sys.argv[3]
	extract(failure_file_name, filter_file_name, output_file_name)
else:
	print ("ERROR, usage: %s <failure log file> <filtered failure file> <output file>" % sys.argv[0])
	print ("<failure log file>: dumps from Titan's failure database")
	print ("<filtered failure file>: file with filtered list of failures")
	print ("<output file>: output file with list of failures")

	sys.exit(0)

