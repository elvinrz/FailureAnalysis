
#!/usr/bin/env python

# University of Pittsburgh
# Center for Simulation and Modeling
# Esteban Meneses
# Extracts information about jobs from MOAB logs.
# Date: 03/20/15

import sys
import re
import datetime
import time
import os
import glob
from math import *

# execution time limit on Titan
EXECUTION_LIMIT=1460

### CLASSES ###

class Job:
	""" This class represents a job and its attributes """
	def __init__(self, date, nodes, tasks, wallclockTime, waitTime, execution_time):
		self.date = date
		self.nodes = nodes
		self.tasks = tasks
		self.wallclockTime = wallclockTime
		self.waitTime = waitTime
		self.execution_time = execution_time

### FUNCTIONS ###
def check_string(substring, string):
    if substring in string:
        return True
    return False
	
def generate_jobs(dir_name, outputFileName):
	""" Reads a failure log file and correlates job IDs with MOAB log files in the directory """
	dayFormat = '%a_%b_%d_%Y'
	file_count = 0
	job_count = 0
	count_event = 0
	cancelled_jobs = 0
	jobs = {}
	pathFileName = []
	

	# start timer
	startTime = time.clock()
    #get all files of the year
	for path, dirs, files in os.walk(dir_name):
		for d in dirs:
			for f in glob.iglob(os.path.join(path, d, '*')):
				pathFileName.append(f)
				
	
	
				
	# going through all files in directory
	for file_name in pathFileName:  
		file_count = file_count + 1
		print ("Progress: %d%%, Analyzing file %s, file_count: %d" % (file_count/365*100, file_name, file_count),end="\r")
		sys.stdout.flush()
		
		conunt_event = 0
		with open(file_name) as log:
			for event in log:
				columns = event.split()
				eventType = columns[2]
				objid = columns[3]
				objEvent = columns[4]
				count_event = count_event + 1
				
				if len(columns) < 6 or eventType != 'job':
					continue													# continue if empty event  # continue if not job event
				
				###########################################################
				if len(columns) > 3 and columns[4] == 'JOBEND':  
					columns_check = event.split()
					columns[5] = 0  #0 because the value is diferent from 2015 value	
					for item in columns_check:
						if  "REQUESTEDTC" in item:
							columns[6] = item[12:]  	#REQUESTEDTC	
							continue
						if "UNAME" in item:							
							columns[7] = item[6:]		#UNAME
							continue
						if "GNAME" in item:							
							columns[8] = item[6:]		#GNAME
							continue
						if "WCLIMIT"in item:	
							columns[9] = item[8:]		#WCLIMIT
							continue
						if "STATE" in item:	
							columns[10] = item[6:]		#STATE
							continue								
						if "SUBMITTIME" in item:
							columns[12] = item[11:]		#SUBMITTIME
							continue
						if "DISPATCHTIME" in item:	
							columns[13] = item[13:]		#DISPATCHTIME
							continue
						if "STARTTIME" in item:	
							columns[14] = item[10:]		#STARTTIME
							continue
						if "COMPLETETIME" in item:	
							columns[15] = item[13:]		#COMPLETETIME
							continue
						if "TASKPERNODE" in item:
							columns[25] = item[12:]		#TASKPERNODE
							continue
						
					job_count = job_count + 1
					# checking cancelled jobs
					
					try:
						dispatch_time = int(columns[13])
					except ValueError:
						dispatch_time = 0
						print("\nAsigned value 0 to dispatch_time")	
					
					try:
						start_time = int(columns[14]) 
					except ValueError:
						start_time = 0
						print("\nAsigned value 0 to start_time")

					if(dispatch_time == 0):
						if(start_time == 0):
							cancelled_jobs = cancelled_jobs + 1
							continue
						else:
							dispatch_time = start_time

					# checking number of requested nodes
					nodes_req = int(columns[5])
					try:
						tasks_req = int(columns[6])
					except ValueError:
						tasks_req = 0
						print("\nAsigned value 0 to tasks_req")
					
					try:
						tasks_per_node = int(columns[25])
					except ValueError:
						tasks_per_node = 0
						print("\nAsigned value 0 to tasks_per_node")
						
					if(nodes_req == 0):
						if(tasks_per_node == -1 or tasks_per_node == 0):
							nodes_req = int(ceil(tasks_req/16.0))
						else:
							nodes_req = int(ceil(tasks_req/float(tasks_per_node)))
					wallclock_req = int(columns[9])/60.0							# transforming wallclock time into minutes
					try:
						submit_time = int(columns[12])
					except ValueError:
						submit_time = 0
						print("\nAsigned value 0 to submit_time")
							
					wait_time = (dispatch_time - submit_time)/60.0					# transforming wait time into minutes
					
					try:
						completion_time = int(columns[15])
					except ValueError:
						completion_time = 0
						print("\nAsigned value 0 to completetion_time")
							
					if(start_time == 0):
						execution_time = (completion_time - dispatch_time)/60.0
					else:
						execution_time = (completion_time - start_time)/60.0			# transforming execution time into minutes
					if(execution_time > EXECUTION_LIMIT):
						print ("--->JOB WITH LONG EXECUTION TIME")
						print ("File: %s, job id: %s, execution time: %f minutes" % (file_name, objid, execution_time))
						execution_time = wallclock_req

					jobs[objid] = Job(file_name[-8:], nodes_req, tasks_req, wallclock_req, wait_time, execution_time)
	print ("\r                                                           ",)

	# creating output file
	outputFile = open(outputFileName, 'w')
	outputFile.write("#jobid\tdate\tnodes\ttasks\twallclockTime\twaitTime\texecutionTime\n")
	for key in jobs.keys():
		outputFile.write("%s\t%s\t%d\t%d\t%d\t%f\t%f\n" % (key, jobs[key].date, jobs[key].nodes, jobs[key].tasks, jobs[key].wallclockTime, jobs[key].waitTime,jobs[key].execution_time))
	outputFile.close()

	# stop timer
	finishTime = time.clock()

	# printing summary
	print ("\nSUMMARY:                                          \n \
	%d files analyzed \n \
	%d Total jobs \n \
	%d jobs analyzed \n \
	%d cancelled jobs \n \
	%.3f seconds execution time" \
	% (file_count, count_event, job_count, cancelled_jobs, finishTime-startTime))

	return

### MAIN CODE ###
if len(sys.argv) >= 3:
	dirName = sys.argv[1]
	outputFileName = sys.argv[2]
	generate_jobs(dirName, outputFileName)
else:
	print ("ERROR, usage: %s <directory> <output file>" % sys.argv[0])
	print ("<directory>: MOAB logs directory")
	print ("<output file>: file name to output job information")
	sys.exit(0)

