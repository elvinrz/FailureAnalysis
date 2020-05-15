#!/usr/bin/env python

# University of Pittsburgh
# Center for Simulation and Modeling
# Esteban Meneses
# Correlates failure events with job events using the job ID as a key.

import sys
import re
import datetime
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
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
	
def failureJob(fileName, dirName, outputFileName, year):
	""" Reads a failure log file and correlates job IDs with MOAB log files in the directory """
	timeFormat = '%m/%d/%y %H:%M %p'
	formatAlt = '%Y-%m-%d %H:%M:%S'
	dayFormat = '%a_%b_%d_%Y'
	count = 0
	missing = 0
	fail_found = 0
	jobs = {}
	jobs_time_execution = {}
	count_ommited = 0
	jobs_nodes_req = {}
	jobs_node_affected = {}
	jobs_node_failure = []
	jobs_node_req_count = []
	
	hour_time_job = []	
	cancelled_jobs = 0
	DaysToSearch = []
	type_of_failure = []
	type_of_failure_11 = []
	type_of_failure_40 = []
	type_of_failure_100 = []
	type_of_failure_600 = []
	
	
	#initialize array of nodes
	for i in range(7):
		jobs_node_failure.append(0)
		jobs_node_req_count.append(0)
		
	job_hours = []
	for j in range(34):
		column = []
		for i in range(24):
			column.append(0)
		job_hours.append(column)	
	
	job_hours_x_6 = []
	for j in range(34):
		column2 = []
		for i in range(4):
			column2.append(0)
		job_hours_x_6.append(column2)	

		
	#initialize for count jobs by time
	for i in range(19):
		jobs_time_execution[i] = 0
		jobs_nodes_req[i] = 0
		jobs_node_affected[i] = 0
		
	jobs_time_execution[20] = 0
	jobs_time_execution[30] = 0
	jobs_time_execution[40] = 0
	jobs_time_execution[50] = 0
	jobs_time_execution[100] = 0
	jobs_time_execution[150] = 0
	jobs_time_execution[200] = 0
	jobs_time_execution[250] = 0
	jobs_time_execution[300] = 0
	jobs_time_execution[350] = 0
	jobs_time_execution[400] = 0
	jobs_time_execution[450] = 0
	jobs_time_execution[500] = 0
	jobs_time_execution[550] = 0
	jobs_time_execution[600] = 0
	# jobs_time_execution[700] = 0
	# jobs_time_execution[800] = 0
	# jobs_time_execution[900] = 0
	
	
	
	jobs_nodes_req[20]  = jobs_node_affected[20]  = 0
	jobs_nodes_req[30]  = jobs_node_affected[30]  = 0
	jobs_nodes_req[40]  = jobs_node_affected[40]  = 0
	jobs_nodes_req[50]  = jobs_node_affected[50]  = 0
	jobs_nodes_req[100] = jobs_node_affected[100] = 0
	jobs_nodes_req[150] = jobs_node_affected[150] = 0
	jobs_nodes_req[200] = jobs_node_affected[200] = 0
	jobs_nodes_req[250] = jobs_node_affected[250] = 0
	jobs_nodes_req[300] = jobs_node_affected[300] = 0
	jobs_nodes_req[350] = jobs_node_affected[350] = 0
	jobs_nodes_req[400] = jobs_node_affected[400] = 0
	jobs_nodes_req[450] = jobs_node_affected[450] = 0
	jobs_nodes_req[500] = jobs_node_affected[500] = 0
	jobs_nodes_req[550] = jobs_node_affected[550] = 0
	jobs_nodes_req[600] = jobs_node_affected[600] = 0
	

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
			text = fields[7].strip()
			
			nodes = fields[9].strip().split()
			
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
			
			flag = False
			# looking for the corresponding log file in the MOAB directory
			for fecha in DaysToSearch:	
				if flag == False:
					jobFileName = dirName + fecha
		
					#for determine if the patch exist and if the file contains data 
					if os.path.isdir(jobFileName[:-2]) == False:
						print("Patch error")
						continue
					if os.stat(jobFileName).st_size == 0:
						print("File error")
						continue				
					#Progress of excecution
					print ("Progress: %d%%, Failure Analized: %d, Failure found: %d, Count missing ID: %d, Search on: %s "% (count/lines*100, count,fail_found, missing, fecha),end="\r")
					with open(jobFileName) as log:
						if flag == False:
							for event in log:
								columns = event.split()
								job_time = columns[0]
								hour = job_time.split(':')[0]
								eventType = columns[2]
								objid = columns[3]
								objEvent = columns[4]
								
								if len(columns) < 6 or eventType != 'job':# continue if empty event # continue if not job event
									continue														
								if jobid == objid and (columns[4] == 'JOBEND' or  columns[4] == 'JOBCANCEL'):  
									hour_time_job.append(hour)
									###########################################################				
									columns_check = event.split()
									if "STARTTIME" in event:
										for item in columns_check:
											if  "REQUESTEDTC" in item.strip():
												#task_req
												columns[6] = item[12:]  	#REQUESTEDTC	columns[6] == task_req
												columns[5] = ceil(int(item[12:])/16)
												
											if "UNAME" in item:							
												columns[7] = item[6:]		#UNAME
												
											if "GNAME" in item:							
												columns[8] = item[6:]		#GNAME
												
											if "WCLIMIT"in item:	
												#wallclock_req
												columns[9] = item[8:]		#WCLIMIT
												
											if "STATE" in item:	
												columns[10] = item[6:]		#STATE
																				
											if "SUBMITTIME" in item:
												#submit_time
												columns[12] = item[11:]		#SUBMITTIME
												
											if "DISPATCHTIME" in item:	
												#dispatch_time
												columns[13] = item[13:]		#DISPATCHTIME
												
											if "STARTTIME" in item:
												#start_time
												start_time = item[10:]		#STARTTIME									
											if "COMPLETETIME" in item:	
												#completion_time
												complete_time = item[13:]		#COMPLETETIME
												#print("complete time: "+str(columns[15]))
											if "TASKPERNODE" in item:
												#tasks_per_node
												columns[25] = item[12:]		#TASKPERNODE
											else:
												columns[25] = 0
												#continue
									else:			
										try:
											
											wallclock_req = int(columns[9])
											dispatch_time = int(columns[13])
											start_time = columns[14]
											complete_time = columns[15]
										except ValueError:
											dispatch_time = 0
											print("\nAsigned value 0 to dispatch_time")
										
									# try:
										# start_time = int(columns[14]) 
									# except ValueError:
										# start_time = 0
										# print("\nAsigned value 0 to start_time")
										
									# if(dispatch_time == 0):
										# if(start_time == 0):
											# cancelled_jobs = cancelled_jobs + 1
											# print("dddddddddddddddddddddddd")
											# continue
										# else:
											# dispatch_time = start_time
											
									# checking number of requested nodes
									try:
										nodes_req = int(columns[5])
										#print("Nodes req: "+str(nodes_req))
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
										#print("Task per node: "+str(tasks_per_node))
									except ValueError:
										tasks_per_node = 0
										print("\nAsigned value 0 to tasks_per_node")
										
									if(nodes_req == 0):
										if(tasks_per_node == -1 or tasks_per_node == 0):
											nodes_req = int(ceil(tasks_req/16.0))
										else:
											nodes_req = int(ceil(tasks_req/float(tasks_per_node)))
									#print("Nodes req 2: "+str(nodes_req))
									wallclock_req = int(columns[9])/60.0							# transforming wallclock time into minutes
									
									try:
										submit_time = int(columns[12])
									except ValueError:
										submit_time = 0
										print("\nAsigned value 0 to submit_time")
									wait_time = 0
									#wait_time = (dispatch_time - submit_time)/60.0					# transforming wait time into minutes
									#print("Wait time: "+str(wait_time))
									# try:
										# completion_time = int(columns[15])
									# except ValueError:
										# completion_time = 0
										# print("\nAsigned value 0 to completetion_time")
									
									# if(start_time == 0):
									# execution_time = (completion_time - dispatch_time)/60.0
									# else:
									try:	
										execution_time = (int(complete_time) - int(start_time)) / 60 # transforming execution time into minutes
									except ValueError:
										print("Value error.  Value ommited: "+ str(count_ommited))
										count_ommited += 1

									
									
								# if int(execution_time) == 0:
									# print("\nentro\nID:"+ jobid + "---star t:"+columns[14]+"node req: "+str(nodes_req)+" / node failed: "+ str(len(nodes)))
									# print("\nentro\nID:"+ jobid + "---complete t:"+columns[15])
									# print("Execution: "+ str(execution_time))
									# print("Execution R: "+ str(int(execution_time)))
									# print("/////////////////////////////////////")
									# #sys.exit()
								
									#COUNT NODES REQ BY JOBS
									if nodes_req == 1:
										jobs_node_req_count[0] += 1
									if nodes_req == 2:
										jobs_node_req_count[1] += 1
									if nodes_req == 3:
										jobs_node_req_count[2] += 1
									if nodes_req == 4:
										jobs_node_req_count[3] += 1
									if nodes_req == 5:
										jobs_node_req_count[4] += 1
									if nodes_req == 6:
										jobs_node_req_count[5] += 1
									if nodes_req > 7:
										jobs_node_req_count[6] += 1
									
									#COUNT NODES AFFECTED BY FAILURES JOBS
									nodes_afected_by_failure = int(len(nodes))
									
									if nodes_afected_by_failure == 1:
										jobs_node_failure[0] += 1
									if nodes_afected_by_failure == 2:
										jobs_node_failure[1] += 1
									if nodes_afected_by_failure == 3:
										jobs_node_failure[2] += 1
									if nodes_afected_by_failure == 4:
										jobs_node_failure[3] += 1
									if nodes_afected_by_failure == 5:
										jobs_node_failure[4] += 1
									if nodes_afected_by_failure == 6:
										jobs_node_failure[5] += 1
									if nodes_afected_by_failure > 7:
										jobs_node_failure[6] += 1
									
									#COUNT JOBS BY EXECUTION TIME
									# parse error number
									if(description == "GPU XID"):
										#get the number
										match = re.search(r"GPU Xid (\d+)", text)
										description = match.group(0)
									if int(execution_time) == 0:
										type_of_failure.append(failure_type +" / "+description)
									
									if int(execution_time) == 11:
										type_of_failure_11.append(failure_type +" / "+description)
									
									if int(execution_time) in jobs_time_execution:
										jobs_time_execution[int(execution_time)] += 1
										if int(execution_time) < 19:
											job_hours[int(execution_time)][int(hour)] += 1
									else:
										if int(execution_time) > 19 and int(execution_time) < 30:
											jobs_time_execution[20] += 1
											job_hours[19][int(hour)] += 1
										if int(execution_time) > 29 and int(execution_time) < 40:
											jobs_time_execution[30] += 1
											job_hours[20][int(hour)] += 1
										if int(execution_time) > 39 and int(execution_time) < 50:
											jobs_time_execution[40] += 1
											job_hours[21][int(hour)] += 1
										if int(execution_time) > 49 and int(execution_time) < 100:
											jobs_time_execution[50] += 1
											job_hours[22][int(hour)] += 1
											type_of_failure_40.append(failure_type +" / "+description)
										if int(execution_time) > 99 and int(execution_time) < 150:
											jobs_time_execution[100] += 1
											type_of_failure_100.append(failure_type +" / "+description)
											job_hours[23][int(hour)] += 1
										if int(execution_time) > 149 and int(execution_time) < 200:
											jobs_time_execution[150] += 1
											job_hours[24][int(hour)] += 1
										if int(execution_time) > 199 and int(execution_time) < 250:
											jobs_time_execution[200] += 1
											job_hours[25][int(hour)] += 1
										if int(execution_time) > 249 and int(execution_time) < 300:
											jobs_time_execution[250] += 1
											job_hours[26][int(hour)] += 1
										if int(execution_time) > 299 and int(execution_time) < 350:
											jobs_time_execution[300] += 1
											job_hours[27][int(hour)] += 1
										if int(execution_time) > 349 and int(execution_time) < 400:
											jobs_time_execution[350] += 1
											job_hours[28][int(hour)] += 1
										if int(execution_time) > 399 and int(execution_time) < 450:
											jobs_time_execution[400] += 1
											job_hours[29][int(hour)] += 1
										if int(execution_time) > 449 and int(execution_time) < 500:
											jobs_time_execution[450] += 1
											job_hours[30][int(hour)] += 1
										if int(execution_time) > 499 and int(execution_time) < 550:
											jobs_time_execution[500] += 1
											job_hours[31][int(hour)] += 1
										if int(execution_time) > 549 and int(execution_time) < 600:
											jobs_time_execution[550] += 1
											job_hours[32][int(hour)] += 1
										if int(execution_time) >= 600:
											jobs_time_execution[600] += 1
											type_of_failure_600.append(failure_type +" / "+description)
											job_hours[33][int(hour)] += 1
										
									if int(execution_time) in jobs_nodes_req:
										jobs_nodes_req[int(execution_time)] += nodes_req
									else:
										if int(execution_time) > 19 and int(execution_time) < 30:
											jobs_nodes_req[20] += nodes_req
										if int(execution_time) > 29 and int(execution_time) < 40:
											jobs_nodes_req[30] += nodes_req
										if int(execution_time) > 39 and int(execution_time) < 50:
											jobs_nodes_req[40] += nodes_req
										if int(execution_time) > 49 and int(execution_time) < 100:
											jobs_nodes_req[50] += nodes_req
										if int(execution_time) > 99 and int(execution_time) < 150:
											jobs_nodes_req[100] += nodes_req
										if int(execution_time) > 149 and int(execution_time) < 200:
											jobs_nodes_req[150] += nodes_req	
										if int(execution_time) > 199 and int(execution_time) < 250:
											jobs_nodes_req[200] += nodes_req
										if int(execution_time) > 249 and int(execution_time) < 300:
											jobs_nodes_req[250] += nodes_req
										if int(execution_time) > 299 and int(execution_time) < 350:
											jobs_nodes_req[300] += nodes_req
										if int(execution_time) > 349 and int(execution_time) < 400:
											jobs_nodes_req[350] += nodes_req
										if int(execution_time) > 399 and int(execution_time) < 450:
											jobs_nodes_req[400] += nodes_req
										if int(execution_time) > 449 and int(execution_time) < 500:
											jobs_nodes_req[450] += nodes_req
										if int(execution_time) > 499 and int(execution_time) < 550:
											jobs_nodes_req[500] += nodes_req
										if int(execution_time) > 549 and int(execution_time) < 600:
											jobs_nodes_req[550] += nodes_req
										if int(execution_time) >= 600:
											jobs_nodes_req[600] += nodes_req
									
									if int(execution_time) in jobs_node_affected:
										jobs_node_affected[int(execution_time)] += int(nodes_afected_by_failure)
									else:
										if int(execution_time) > 19 and int(execution_time) < 30:
											jobs_node_affected[20] += int(nodes_afected_by_failure)
										if int(execution_time) > 29 and int(execution_time) < 40:
											jobs_node_affected[30] += int(nodes_afected_by_failure)
										if int(execution_time) > 39 and int(execution_time) < 50:
											jobs_node_affected[40] += int(nodes_afected_by_failure)
										if int(execution_time) > 49 and int(execution_time) < 100:
											jobs_node_affected[50] += int(nodes_afected_by_failure)
										if int(execution_time) > 99 and int(execution_time) < 150:
											jobs_node_affected[100] += int(nodes_afected_by_failure)
										if int(execution_time) > 149 and int(execution_time) < 200:
											jobs_node_affected[150] += int(nodes_afected_by_failure)	
										if int(execution_time) > 199 and int(execution_time) < 250:
											jobs_node_affected[200] += int(nodes_afected_by_failure)
										if int(execution_time) > 249 and int(execution_time) < 300:
											jobs_node_affected[250] += int(nodes_afected_by_failure)
										if int(execution_time) > 299 and int(execution_time) < 350:
											jobs_node_affected[300] += int(nodes_afected_by_failure)
										if int(execution_time) > 349 and int(execution_time) < 400:
											jobs_node_affected[350] += int(nodes_afected_by_failure)
										if int(execution_time) > 399 and int(execution_time) < 450:
											jobs_node_affected[400] += int(nodes_afected_by_failure)
										if int(execution_time) > 449 and int(execution_time) < 500:
											jobs_node_affected[450] += int(nodes_afected_by_failure)
										if int(execution_time) > 499 and int(execution_time) < 550:
											jobs_node_affected[500] += int(nodes_afected_by_failure)
										if int(execution_time) > 549 and int(execution_time) < 600:
											jobs_node_affected[550] += int(nodes_afected_by_failure)
										if int(execution_time) >= 600:
											jobs_node_affected[600] += int(nodes_afected_by_failure)
																
									
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
				# print ("--------------------------------------")
				# print("Job id: "+jobid)
				# print("Obj id: "+objid)
				# print("data: "+ dateAndTime)
				# print ("--------------------------------------")
				#sys.exit()
		check_output.close()
		# stop timer
		finishTime = time.clock()

		for i in range(34):
			job_hours_x_6[i][0] = job_hours[i][0] + job_hours[i][1] + job_hours[i][2] + job_hours[i][3] + job_hours[i][4] + job_hours[i][5]
			job_hours_x_6[i][1] = job_hours[i][6] + job_hours[i][7] + job_hours[i][8] + job_hours[i][9] + job_hours[i][10] + job_hours[i][11]
			job_hours_x_6[i][2] = job_hours[i][12] + job_hours[i][13] + job_hours[i][14] + job_hours[i][15] + job_hours[i][16] + job_hours[i][17]
			job_hours_x_6[i][3] = job_hours[i][18] + job_hours[i][19] + job_hours[i][20] + job_hours[i][21] + job_hours[i][22] + job_hours[i][23]
		
		
		data1 = [job_hours_x_6[0][0], job_hours_x_6[1][0], job_hours_x_6[2][0], job_hours_x_6[3][0], job_hours_x_6[4][0], job_hours_x_6[5][0], \
				job_hours_x_6[6][0], job_hours_x_6[7][0], job_hours_x_6[8][0], job_hours_x_6[9][0],	job_hours_x_6[10][0], job_hours_x_6[11][0], \
				job_hours_x_6[12][0], job_hours_x_6[13][0], job_hours_x_6[14][0], job_hours_x_6[15][0], job_hours_x_6[16][0], job_hours_x_6[17][0], \
				job_hours_x_6[18][0], job_hours_x_6[19][0], job_hours_x_6[20][0], job_hours_x_6[21][0], job_hours_x_6[22][0], job_hours_x_6[23][0], \
				job_hours_x_6[24][0], job_hours_x_6[25][0], job_hours_x_6[26][0], job_hours_x_6[27][0], job_hours_x_6[28][0], job_hours_x_6[29][0],  \
				job_hours_x_6[30][0], job_hours_x_6[31][0], job_hours_x_6[32][0], job_hours_x_6[33][0]]
				
		data2 = [job_hours_x_6[0][1], job_hours_x_6[1][1], job_hours_x_6[2][1], job_hours_x_6[3][1], job_hours_x_6[4][1], job_hours_x_6[5][1], \
				job_hours_x_6[6][1], job_hours_x_6[7][1], job_hours_x_6[8][1], job_hours_x_6[9][1],	job_hours_x_6[10][1], job_hours_x_6[11][1], \
				job_hours_x_6[12][1], job_hours_x_6[13][1], job_hours_x_6[14][1], job_hours_x_6[15][1], job_hours_x_6[16][1], job_hours_x_6[17][1], \
				job_hours_x_6[18][1], job_hours_x_6[19][1], job_hours_x_6[20][1], job_hours_x_6[21][1], job_hours_x_6[22][1], job_hours_x_6[23][1], \
				job_hours_x_6[24][1], job_hours_x_6[25][1], job_hours_x_6[26][1], job_hours_x_6[27][1], job_hours_x_6[28][1], job_hours_x_6[29][1],  \
				job_hours_x_6[30][1], job_hours_x_6[31][1], job_hours_x_6[32][1], job_hours_x_6[33][1]]
				
		data3 = [job_hours_x_6[0][2], job_hours_x_6[1][2], job_hours_x_6[2][2], job_hours_x_6[3][2], job_hours_x_6[4][2], job_hours_x_6[5][2], \
				job_hours_x_6[6][2], job_hours_x_6[7][2], job_hours_x_6[8][2], job_hours_x_6[9][2],	job_hours_x_6[10][2], job_hours_x_6[11][2], \
				job_hours_x_6[12][2], job_hours_x_6[13][2], job_hours_x_6[14][2], job_hours_x_6[15][2], job_hours_x_6[16][2], job_hours_x_6[17][2], \
				job_hours_x_6[18][2], job_hours_x_6[19][2], job_hours_x_6[20][2], job_hours_x_6[21][2], job_hours_x_6[22][2], job_hours_x_6[23][2], \
				job_hours_x_6[24][2], job_hours_x_6[25][2], job_hours_x_6[26][2], job_hours_x_6[27][2], job_hours_x_6[28][2], job_hours_x_6[29][2],  \
				job_hours_x_6[30][2], job_hours_x_6[31][2], job_hours_x_6[32][2], job_hours_x_6[33][2]]
				
		data4 = [job_hours_x_6[0][3], job_hours_x_6[1][3], job_hours_x_6[2][3], job_hours_x_6[3][3], job_hours_x_6[4][3], job_hours_x_6[5][3], \
				job_hours_x_6[6][3], job_hours_x_6[7][3], job_hours_x_6[8][3], job_hours_x_6[9][3],	job_hours_x_6[10][3], job_hours_x_6[11][3], \
				job_hours_x_6[12][3], job_hours_x_6[13][3], job_hours_x_6[14][3], job_hours_x_6[15][3], job_hours_x_6[16][3], job_hours_x_6[17][3], \
				job_hours_x_6[18][3], job_hours_x_6[19][3], job_hours_x_6[20][3], job_hours_x_6[21][3], job_hours_x_6[22][3], job_hours_x_6[23][3], \
				job_hours_x_6[24][3], job_hours_x_6[25][3], job_hours_x_6[26][3], job_hours_x_6[27][3], job_hours_x_6[28][3], job_hours_x_6[29][3],  \
				job_hours_x_6[30][3], job_hours_x_6[31][3], job_hours_x_6[32][3], job_hours_x_6[33][3]]
		
		print(data1)
		print(data2)
		print(data3)
		print(data4)
		###########################################################################################
		###########################################################################################
			
		if "2015" in dirName:
			data_total_jobs = [95099, 23393, 14805, 9969, 6811, 5940, 5104, 5040, 3306, 2574, 3104, 3038, 4551, 2676, 1910, 1892, 1902, 2052, 1750, 11080, 9966, 8527, 46386, 50559, 2386, 3102, 1405, 1943, 3546, 282, 321, 269, 210, 9408]
			show_bar_count = [95099, 23393, 14805, 9969, 6811, "", "", "", "", "", "", "", "", "", "", "", "", "", "", 11080, "", "", 46386, 50559, "", "", "", "", "", "", "", "", "", 9408]
		
		if "2016" in dirName:
			data_total_jobs = [109763, 26277, 15057, 10175, 7714, 6565, 5043, 4563, 3532, 3000, 3772, 3073, 6433, 5059, 3045, 2682, 1904, 1547, 1637, 16286, 13510, 13664, 97863, 61350, 2223, 2679, 1838, 1923, 3403, 236, 207, 251, 172, 4768]
			show_bar_count = [109763, 26277, 15057, 10175, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 16286, "", "", 97863, 61350, "", "", "", "", "", "", "", "", "", 4768]
		#PLOT THE DATA 
		fig = plt.figure()
		fig, axs = plt.subplots(2,2,figsize=(8, 5))
		
		print("\nProcessing plots 1")
		
		data = []
		for key in sorted(jobs_time_execution.keys()):	
			data.append(jobs_time_execution[key])
		
		print(data)
		print("jobs node affected----------------------------------")
		print(jobs_time_execution)
		print("-------------------------------------------------------------------------------------------------------")
		median = np.median(data)
		print("Median: "+str(median))
		print("-------------------------------------------------------------------------------------------------------")
		
		
		axs[0,0].bar(range(0,34,1), data,color=['blue'],width=.5)
		axs[0,0].tick_params(axis = 'x',  labelsize = 6.3)
		axs[0,0].tick_params(axis = 'y',  labelsize = 7)
		axs[0,0].set_xlim(-1,35, auto=False)
		axs[0,0].set_xlabel('Execution Time (Minutes)')
		axs[0,0].set_ylabel('Jobs')
		axs[0,0].set_title('')
		axs[0,0].set_xticklabels(['0','','2','','4','','6','','8','','10','','12','','14','','16','','18','','20+','','40+','','100+','','200+','','300+','','400+','','500+','','600+'],rotation = 45)
		axs[0,0].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34])
		#axs[0,0].legend(edgecolor="black",prop={'size': 7})
		
		#execution time with job hour
		axs[1,0].bar(range(0,34,1), data1,  color=['cyan'],width=.5, label = "0-5 hours", edgecolor = "black", linewidth = 0.1)
		axs[1,0].bar(range(0,34,1), data2, bottom=data1,color=['green'],width=.5, label = "6-11 hours", edgecolor = "black", linewidth = 0.1)
		axs[1,0].bar(range(0,34,1), data3, bottom=np.array(data1) + np.array(data2),color=['magenta'],width=.5, label = "12-17 hours", edgecolor = "black", linewidth = 0.1)
		axs[1,0].bar(range(0,34,1), data4, bottom=np.array(data1) + np.array(data2) + np.array(data3),color=['red'],width=.5, label = "18-23 hours", edgecolor = "black", linewidth = 0.1)
		
		axs[1,0].tick_params(axis = 'x',  labelsize = 6.3)
		axs[1,0].tick_params(axis = 'y',  labelsize = 7)
		axs[1,0].set_xlim(-1,35, auto=False)
		axs[1,0].set_xlabel('Execution Time (Minutes)')
		axs[1,0].set_ylabel('Jobs')
		axs[1,0].set_title('')
		axs[1,0].set_xticklabels(['0','','2','','4','','6','','8','','10','','12','','14','','16','','18','','20+','','40+','','100+','','200+','','300+','','400+','','500+','','600+'],rotation = 45)
		axs[1,0].set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34])
		axs[1,0].legend(edgecolor="black",prop={'size': 5})
		
		##################################################################
		
		barWidth = 0.35
		r = np.arange(0,34,1)
		r1 = [x - 0.2 + barWidth for x in r]
		r2 = [x + 0.007 + barWidth for x in r]
	
		axs[1,1].set_xlabel('Execution Time (Minutes)')
		axs[1,1].set_title('')
		ax = plt.gca()
		#axs[1,1].bar(r1,data,width=barWidth, label="Failed Jobs",color=['blue'], align="edge")
		axs[1,1].bar(r1, data1, width=barWidth, color=['cyan'], label = "0-5 hours", edgecolor = "black", linewidth = 0.1)
		axs[1,1].bar(r1, data2, width=barWidth, bottom=data1,color=['green'], label = "6-11 hours", edgecolor = "black", linewidth = 0.1)
		axs[1,1].bar(r1, data3, width=barWidth, bottom=np.array(data1) + np.array(data2),color=['magenta'],label = "12-17 hours", edgecolor = "black", linewidth = 0.1)
		axs[1,1].bar(r1, data4, width=barWidth, bottom=np.array(data1) + np.array(data2) + np.array(data3),color=['red'], label = "18-23 hours", edgecolor = "black", linewidth = 0.1)
		#axs[1,1].legend(edgecolor="black", prop={'size': 5})
		axs[1,1].set_xlim(-1,35, auto=False)
		
		axs[1,1].tick_params(axis = 'x',  labelsize = 6.3)
		axs[1,1].set_ylim([1,max(data)+10])
		axs[1,1].set_ylabel('Failed Jobs')
		axs[1,1].tick_params(axis = 'y', labelsize = 7)
		
		axs2 = axs[1,1].twinx()
		axs2.bar(r2,data_total_jobs,width=barWidth, label="Total Jobs",color=['blue'], align="edge")
		axs2.set_ylim([1,max(data_total_jobs)+5000])
		axs2.set_ylabel('Total Jobs',color='tab:blue' )
		axs2.tick_params(axis = 'y',  labelsize = 7, labelcolor='tab:blue')
		axs2.legend(edgecolor="black", prop={'size': 5})
		
		
		for i in range(len(r2)):
			try:
				axs2.text(x = r2[i] , y = show_bar_count[i]+1000, s = show_bar_count[i], size = 3)
			except:
				print("")
		#axs2.legend(edgecolor="black",prop={'size': 8},bbox_to_anchor=(0,1.02,1,0.2), loc="lower center", borderaxespad=0, ncol=2)
		
		axs[1,1].set_xlim(-1,35, auto=False)
		axs[1,1].set_xticklabels(['0','','2','','4','','6','','8','','10','','12','','14','','16','','18','','20+','','40+','','100+','','200+','','300+','','400+','','500+','','600+'],rotation = 45)
		axs[1,1].set_xticks([0.32,1.32,2.32,3.32,4.32,5.32,6.32,7.32,8.32,9.32,10.32,11.32,12.32,13.32,14.32,15.32,16.32,17.32,18.32,19.32,20.32,21.32,22.32,23.32,24.32,25.32,26.32,27.32,28.32,29.32,30.32,31.32,32.32,33.32,34.32])
		#axs[1,1].legend(edgecolor="black",prop={'size': 8},bbox_to_anchor=(0,1.02,1,0.2), loc="lower center", borderaxespad=0, ncol=2)
		
		###########################################################################################
		###########################################################################################
		print("\nProcessing plots 2")
		
		fail_node = []
		for key in sorted(jobs_node_affected.keys()):	
			fail_node.append(jobs_node_affected[key])

		
		node_req = []
		for key in sorted(jobs_nodes_req.keys()):	
			node_req.append(jobs_nodes_req[key])
		
		barWidth = 0.35
		r1 = np.arange(0,34,1)
		r2 = [x + 0.01 + barWidth for x in r1]
	
		axs[0,1].set_xlabel('Execution Time (Minutes)')
		axs[0,1].set_ylabel('Node Count')
		axs[0,1].set_title('')
		ax = plt.gca()
		axs[0,1].bar(r1,fail_node,width=barWidth, label="Failed Nodes",color=['blue'], align="edge",log = True)
		axs[0,1].bar(r2,node_req,width=barWidth, label="Requested Nodes",color=['red'], align="edge", log = True)
		axs[0,1].set_ylim([1,max(node_req)+500])
		
		axs[0,1].tick_params(axis = 'x', labelsize = 6.3)
		axs[0,1].tick_params(axis = 'y',  labelsize = 7)
		
		axs[0,1].set_xlim(-1,35, auto=False)
		axs[0,1].set_xticklabels(['0','','2','','4','','6','','8','','10','','12','','14','','16','','18','','20+','','40+','','100+','','200+','','300+','','400+','','500+','','600+'],rotation = 45)
		axs[0,1].set_xticks([0.3,1.3,2.3,3.3,4.3,5.3,6.3,7.3,8.3,9.3,10.3,11.3,12.3,13.3,14.3,15.3,16.3,17.3,18.3,19.3,20.3,21.3,22.3,23.3,24.3,25.3,26.3,27.3,28.3,29.3,30.3,31.3,32.3,33.3,34.3])
		axs[0,1].legend(edgecolor="black",prop={'size': 8},bbox_to_anchor=(0,1.02,1,0.2), loc="lower center", borderaxespad=0, ncol=2)
		
		plt.subplots_adjust(top=0.92, bottom=0.2, left=0.05, right=0.90, hspace=0.50, wspace=0.3)
		plt.savefig("PLOT_count_jobs_by_time_execution_"+year+".pdf")
		print("\nPLOT_count_jobs_by_time_execution_"+year+".pdf>")
		
		plt.clf() 
		fig = plt.figure()
		fig, axs = plt.subplots(2,3,figsize=(15, 8))
		
		print("\nProcessing plots 3")
		print(jobs_node_failure)
		print("------------------------------------------")
		print(jobs_node_req_count)
		
		r1 = np.arange(1,8,1)
		r2 = [x + 0.02 + barWidth for x in r1]
		barWidth = 0.3
		axs[0,0].bar(r1,jobs_node_failure,width=barWidth,label="Failure nodes",color=['blue'],log=True, align="edge")
		axs[0,0].bar(r2,jobs_node_req_count,width=barWidth,label="Requested nodes",color=['red'],log=True,align="edge")
		axs[0,0].set_xlabel('Nodes Affected')
		axs[0,0].set_ylabel('Jobs count')
		axs[0,0].set_title('')
		axs[0,0].set_xticklabels(['1','2','3','4','5','6','>7'])
		axs[0,0].set_xticks([1.15,2.15,3.15,4.15,5.15,6.15,7.15])
		axs[0,0].legend(framealpha=1,shadow=True, borderpad = 1, fancybox=True)

		
		plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.95, hspace=0.25, wspace=0.25)
		plt.savefig("PLOT_count_nodes_affected_by_failure_jobs_"+year+".pdf")
		print("\nPLOT_count_nodes_affected_by_failure_jobs.pdf_"+year+".pdf>")
		# creating output file
		outputFile = open(outputFileName, 'w')
		outputFile.write("jobid\tdate\tnodes req\ttasks req\twallclockTime\twaitTime\texecutionTime\n")
		for key in jobs.keys():
			outputFile.write("%s\t%s\t%d\t%d\t%d\t%f\t%f\t%s\n" % (key, jobs[key].date, jobs[key].nodes, jobs[key].tasks, jobs[key].wallclockTime, jobs[key].waitTime,jobs[key].execution_time, failure_type))
		outputFile.close()
	
		# printing summary
		print ("\nSUMMARY:\n\t%d failures analyzed \n\t%d job ids missing \n\t%.2f %% correlation reliability\n\t%.3f seconds execution time" % (count, missing, (count-missing)/float(count)*100.0, finishTime-startTime))
	
		print("----------------------------")
		print("Failures range 0")
		for c in type_of_failure:
			print(c)
		print("----------------------------")
		print("Failures range 11")
		for c in type_of_failure_11:
			print(c)
		print("----------------------------")
		print("Failures range 40+")
		for c in type_of_failure_40:
			print(c)
		print("----------------------------")
		print("Failures range 100+")
		for c in type_of_failure_100:
			print(c)
		print("----------------------------")
		print("Failures range 600+")
		for c in type_of_failure_600:
			print(c)
	
		print("----------------------------")
		print(job_hours)
		print("----------------------------")
		
		print("failed nodes")
		print(jobs_node_affected)
		for c in jobs_node_affected:
			print(jobs_node_affected[c])
		print("----------------------------")
		
		
		print("requested nodes")
		print(jobs_nodes_req)
		for c in jobs_nodes_req:
			print(jobs_nodes_req[c])
		print("----------------------------")
		
	return

### MAIN CODE ###
if len(sys.argv) >= 5:
	fileName = sys.argv[1]
	dirName = sys.argv[2]
	outputFileName = sys.argv[3]
	year = sys.argv[4]
	failureJob(fileName, dirName, outputFileName, year)
else:
	print ("ERROR, usage: %s <file> <directory> <output file>" % sys.argv[0])
	print ("<file>: failure log file")
	print ("<directory>: MOAB logs directory")
	print ("<output file>: file name to output failed-job information")
	print ("<year>: year for data processing")
	
	sys.exit(0)

