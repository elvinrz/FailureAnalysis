#!/usr/bin/env python

# University of Pittsburgh
# Center for Simulation and Modeling
# Esteban Meneses
# Correlates failure events with job events using the job ID as a key.

import sys
import re
import datetime
import time
import matplotlib.pyplot as plt
import os
from math import *

# execution time limit on Titan
EXECUTION_LIMIT=1450

### CLASSES ###

class Job:
	""" This class represents a failed job and its attributes """
	def __init__(self, date, nodes, tasks, wallclockTime, waitTime, execution_time, failure_type):
		self.date = date
		self.nodes = nodes
		self.tasks = tasks
		self.wallclockTime = wallclockTime
		self.waitTime = waitTime
		self.execution_time = execution_time
		self.failure_type = failure_type

### FUNCTIONS ###
def check_string(substring, string):
    if substring in string:
        return True
    return False
	
def failureJob(fileName, dirName, outputFileName):
	""" Reads a failure log file and correlates job IDs with MOAB log files in the directory """
	timeFormat = '%m/%d/%y %H:%M %p'
	formatAlt = '%Y-%m-%d %H:%M:%S'
	dayFormat = '%a_%b_%d_%Y'
	count = 0
	missing = 0
	fail_found = 0
	jobs = {}
	cancelled_jobs = 0
	DaysToSearch = []

	# start timer
	startTime = time.clock()
	
	check_output = open("check_output", 'w')
						
	#size of failure file
	with open(fileName) as f:
		lines = len(f.readlines())
	
	# going through all entries in the file
	with open(fileName) as f:
		next(f)  # skipping first line (header)
		for line in f:
			# getting jobid and time for each failure
			count = count + 1
			fields = line.split('|')													# separating fields
			jobid = fields[2].strip()													# reading job ib
			dateAndTime = fields[3].strip()												# reading time
			failure_type = fields[4].strip()
			description  = fields[6].strip()
			print("jobid:"+jobid+" time:"+dateAndTime+" type:"+failure_type+"desc: "+description)
			sys.exit()
			try:
				currentDate = datetime.datetime.strptime(dateAndTime, timeFormat)
			except ValueError:
				try:
					currentDate = datetime.datetime.strptime(dateAndTime, formatAlt)
				except ValueError:					
					print ("ERROR with date ", dateAndTime)
					sys.exit(0)

			DaysToSearch.append(str(currentDate.month).zfill(2) +"/"+ str(currentDate.day).zfill(2))
			
			oneDayBefore = currentDate - datetime.timedelta(days=1)
			DaysToSearch.append(str(oneDayBefore.month).zfill(2)+"/"+str(oneDayBefore.day).zfill(2))
			
			twoDayBefore = currentDate - datetime.timedelta(days=2)
			DaysToSearch.append(str(twoDayBefore.month).zfill(2)+"/"+str(twoDayBefore.day).zfill(2))
			
			oneDayAfter = currentDate + datetime.timedelta(days=1)
			DaysToSearch.append(str(oneDayAfter.month).zfill(2)+"/"+str(oneDayAfter.day).zfill(2))
			
			twoDayAfter = currentDate + datetime.timedelta(days=2)
			DaysToSearch.append(str(twoDayAfter.month).zfill(2)+"/"+str(twoDayAfter.day).zfill(2))
			

			# looking for the corresponding log file in the MOAB directory
			for fecha in DaysToSearch:	
				jobFileName = dirName + fecha
	
				#for determine if the patch exist and if the file contains data 
				if os.path.isdir(jobFileName[:-2]) == False:
					continue
				if os.stat(jobFileName).st_size == 0:
					continue
				
				#Progress of excecution
				print ("Progress: %d%%, Failure Analized: %d, Failure found: %d, Count missing ID: %d, Search on: %s "% (count/lines*100, count,fail_found, missing, jobFileName),end="\r") 

				with open(jobFileName) as log:
					flag = False
						
					for event in log:
						columns = event.split()
						eventType = columns[2]
						objid = columns[3]
						objEvent = columns[4]
						if len(columns) < 6 or eventType != 'job':# continue if empty event # continue if not job event
							continue														
						if jobid == objid and (columns[4] == 'JOBEND' or columns[4] == 'JOBCANCEL'):  
							###########################################################				
							columns_check = event.split()
							columns[5] = 0  #0 because the value is diferent from 2015 value
							
							if "2016" in dirName:	
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
									else:
										columns[25] = 0
										continue
						
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
							try:
								nodes_req = int(columns[5])
							except ValueError:
								nodes_req = 0
								print("\nAsigned value 0 to nodes_req")
								
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
								print ("File: %s, job id: %s, execution time: %f minutes" % (jobFileName[2:], objid, execution_time))
								execution_time = wallclock_req
							flag = True
							fail_found = fail_found + 1
							DaysToSearch.clear()
							#print("Encontrado")
							if not jobid in jobs:
								jobs[objid] = Job(jobFileName[-5:], nodes_req, tasks_req, wallclock_req, wait_time, execution_time, failure_type)
							break
					if not flag:
						missing = missing + 1
	
		check_output.close()
	
		# stop timer
		finishTime = time.clock()

		# creating output file
		outputFile = open(outputFileName, 'w')
		outputFile.write("#jobid\tdate\tnodes\ttasks\twallclockTime\twaitTime\texecutionTime\n")
		for key in jobs.keys():
			outputFile.write("%s\t%s\t%d\t%d\t%d\t%f\t%f\t%s\n" % (key, jobs[key].date, jobs[key].nodes, jobs[key].tasks, jobs[key].wallclockTime, jobs[key].waitTime,jobs[key].execution_time,jobs[key].failure_type))
		outputFile.close()
	
		# printing summary
		print ("\nSUMMARY:\n\t%d failures analyzed \n\t%d job ids missing \n\t%.2f %% correlation reliability\n\t%.3f seconds execution time" % (count, missing, (count-missing)/float(count)*100.0, finishTime-startTime))
	
	return

### MAIN CODE ###
if len(sys.argv) >= 4:
	fileName = sys.argv[1]
	dirName = sys.argv[2]
	outputFileName = sys.argv[3]
	failureJob(fileName, dirName, outputFileName)
else:
	print ("ERROR, usage: %s <file> <directory> <output file>" % sys.argv[0])
	print ("<file>: failure log file")
	print ("<directory>: MOAB logs directory")
	print ("<output file>: file name to output failed-job information")
	sys.exit(0)

