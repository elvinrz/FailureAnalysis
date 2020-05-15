#!/usr/bin/env python

# Costa Rica High Technology Center (CeNAT)
# Advanced Computing Laboratory
# Esteban Meneses, PhD (esteban.meneses@acm.org)
# Correlates failure events with job events using the job ID as a key.

import sys
import re
import datetime
import time
import os


### FUNCTIONS ###

def correlateFailureJob(fileName, dirName, reportName):
	""" Reads a failure log file and correlates job IDs with MOAB log files in the directory """
	timeFormat = '%m/%d/%y %H:%M %p'
	formatAlt = '%Y-%m-%d %H:%M:%S'
	dayFormat = '%a_%b_%d_%Y'
	count = 0
	missing = 0
	
	DaysToSearch = []

	
	# open output file
	outputFile = open(reportName, 'w')
    
	# start timer
	startTime = time.clock()
	
	#size of failure file
	with open(fileName) as f:
		lines = len(f.readlines())

	# going through all entries in the file
	with open(fileName) as f:
		next(f)																			# skipping first line (header)
		for line in f:
			jobFileName = ""
			DaysToSearch.clear()
			# getting jobid and time for each failure
			count = count + 1
			fields = line.split('|')													# separating fields
			jobid = fields[2].strip()													# reading job ib
			#print ("job id %s",jobid)
			
			dateAndTime = fields[3].strip()												# reading time
			#print ("fecha %s",dateAndTime)
			try:
				currentDate = datetime.datetime.strptime(dateAndTime, timeFormat)
			except ValueError:
				try:
					currentDate = datetime.datetime.strptime(dateAndTime, formatAlt)
				except ValueError:					
					print("ERROR with date %s in line %s" % (dateAndTime,line))
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
			
			#looking for the corresponding log file in the MOAB directory
			for fecha in DaysToSearch:
				jobFileName = dirName + fecha
				
				#Progress of excecution
				
				print ("Progress: %d%%, Failure Analized: %d, Count missing ID: %d, Search on: %s "% (count/lines*100, count, missing, jobFileName),end="\r") 
				sys.stdout.flush()
				
				#for determine if the patch exist and if the file contains data 
				if os.path.isdir(dirName[:-1]) == False:
					continue
				if os.stat(jobFileName).st_size == 0:
				    continue
					
				with open(jobFileName) as log:
					flag = False
					next(log)
					for event in log:
						columns = event.split()
						if len(columns) < 6:
							continue														# continue if empty event
						eventType = columns[2]
						if eventType != 'job':
							continue														# continue if not job event
						objid = columns[3]
						event_type = columns[4]
						nodes = columns[5]
						if len(columns) > 3 and jobid == objid and (event_type == 'JOBEND' or event_type == 'JOBCANCEL'):
							flag = True
							DaysToSearch.clear()
							break	
							
					if not flag:
						outputFile.write("Job ID %s with date %s is not found in MOAB logs \n" % (jobid,DaysToSearch[0]))
						missing = missing + 1
									
		# close output file
		outputFile.close()

		# stop timer
		finishTime = time.clock()
	
		# printing summary
		print("\nSUMMARY:\n\t%d failures analyzed \n\t%d job ids missing \n\t%.2f %% correlation reliability\n\t%.3f seconds execution time \nReport in file %s" % (count, missing, (count-missing)/float(count)*100.0, finishTime-startTime, reportName))
	
	return

### MAIN CODE ###
if len(sys.argv) >= 3:
	fileName = sys.argv[1]
	dirName = sys.argv[2]
	if len(sys.argv) == 4:
		reportName = sys.argv[3]
	else:
		reportName = 'log.txt'
	correlateFailureJob(fileName, dirName, reportName)
else:
	print("ERROR, usage: %s <file> <directory> [<report>]\n<file>: failure log file\n<directory>: MOAB logs directory by year(/titan_data/20XX/)\n<report>: output file with stats about the analysis; by default it is named log.txt" % sys.argv[0])
	sys.exit(0)

