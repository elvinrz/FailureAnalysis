#!/usr/bin/env python

# University of Pittsburgh
# Center for Simulation and Modeling
# Esteban Meneses
# Extracts information about jobs from MOAB logs.
# Date: 04/04/15

import sys
import re
# import datetime
import time
import os
import math as mt
# import glob
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
# import numpy as np
# from math import *
# import calendar as cl
# import datetime

# from datetime import date, timedelta, datetime as dt

### FUNCTIONS ###


def init_tables(hour_table_jobs_all_run, hour_table, day_table, month_table):
	""" Initializes tables """
		
	hour_table_jobs_all_run[0] = 0
	hour_table_jobs_all_run[1] = 0
	hour_table_jobs_all_run[2] = 0
	hour_table_jobs_all_run[3] = 0
	hour_table_jobs_all_run[4] = 0
	hour_table_jobs_all_run[5] = 0
	hour_table_jobs_all_run[6] = 0
	hour_table_jobs_all_run[7] = 0
	hour_table_jobs_all_run[8] = 0
	hour_table_jobs_all_run[9] = 0
	hour_table_jobs_all_run[10] = 0
	hour_table_jobs_all_run[11] = 0
	hour_table_jobs_all_run[12] = 0
	hour_table_jobs_all_run[13] = 0
	hour_table_jobs_all_run[14] = 0
	hour_table_jobs_all_run[15] = 0
	hour_table_jobs_all_run[16] = 0
	hour_table_jobs_all_run[17] = 0
	hour_table_jobs_all_run[18] = 0
	hour_table_jobs_all_run[19] = 0
	hour_table_jobs_all_run[20] = 0
	hour_table_jobs_all_run[21] = 0
	hour_table_jobs_all_run[22] = 0
	hour_table_jobs_all_run[23] = 0
	hour_table['00'] = 0
	hour_table['01'] = 0
	hour_table['02'] = 0
	hour_table['03'] = 0
	hour_table['04'] = 0
	hour_table['05'] = 0
	hour_table['06'] = 0
	hour_table['07'] = 0
	hour_table['08'] = 0
	hour_table['09'] = 0
	hour_table['10'] = 0
	hour_table['11'] = 0
	hour_table['12'] = 0
	hour_table['13'] = 0
	hour_table['14'] = 0
	hour_table['15'] = 0
	hour_table['16'] = 0
	hour_table['17'] = 0
	hour_table['18'] = 0
	hour_table['19'] = 0
	hour_table['20'] = 0
	hour_table['21'] = 0
	hour_table['22'] = 0
	hour_table['23'] = 0
	day_table['Mon'] = 0
	day_table['Tue'] = 0
	day_table['Wed'] = 0
	day_table['Thu'] = 0
	day_table['Fri'] = 0
	day_table['Sat'] = 0
	day_table['Sun'] = 0
	month_table['Jan'] = 0
	month_table['Feb'] = 0
	month_table['Mar'] = 0
	month_table['Apr'] = 0
	month_table['May'] = 0
	month_table['Jun'] = 0
	month_table['Jul'] = 0
	month_table['Aug'] = 0
	month_table['Sep'] = 0
	month_table['Oct'] = 0
	month_table['Nov'] = 0
	month_table['Dec'] = 0
	
	
def generate(dir_name, output_dir_name):
	#""" Reads a failure log file and correlates job IDs with MOAB log files in the directory """
	timeFormat = '%m/%d/%y %H:%M %p'
	formatAlt = '%Y-%m-%d %H:%M:%S'
	job_total_count = 0
	file_count = 0
	line_count = 0
	job_count = 0
	hour_table = {}
	day_table = {}
	month_table = {}
	pathFileName = []
	hour_table_jobs_all_run = {}
	
	
	init_tables(hour_table_jobs_all_run,hour_table,day_table,month_table)

	# start timer
	startTime = time.clock()
	
	#get all files of the year
	for path, dirs, files in os.walk(dir_name):
			for f in files:
				pathFileName.append(f)
	
	
	sizeList = len(pathFileName)
	# # going through all files in directory
	for file_name in pathFileName:
		file_count = file_count + 1

		
		#extract day, day_of_week month
		date_file = file_name.split(".")
		day_of_week = date_file[1][:3]
		day = date_file[1][8:10]
		month = date_file[1][4:7]
		year = date_file[1][-4:]
		
		# print(date_file)
		# print(day_of_week)
		# print(day)
		# print(month)
		# print(year)
		
		job_file_name = dir_name + file_name
		
		with open(job_file_name) as log:
			
			line_count = 0 
			try:
				for event in log:
					job_total_count += 1
					
					line_count += 1
					columns = event.split()
					print ("Progress: %d%%, line: %d, file: %s"% (file_count/sizeList*100, line_count, file_name),end="\r") 
					sys.stdout.flush()
			
					if len(columns) < 6:
						continue														# continue if empty event
					eventType = columns[2]
					# continue if not job event
					if eventType != 'job':
						continue
					hour = columns[0].strip()[:2]
					end_hour = int(hour)
					objid = columns[3]
					objEvent = columns[4]
					
					if len(columns) > 3 and objEvent == 'JOBEND':
				
						##############################################################################
						##############################################################################
						##############################################################################
						##############################################################################
						
						start_time = columns[14]
						complete_time = columns[15]

						# total job execution time
						total_time_hours = mt.ceil((int(complete_time) - int(start_time)) / 60 /60)
										
						#adjust hours < 0 and hour > 23
						if total_time_hours > 48:
							continue
							
						print("___________________________________________________________")
						print("ID: "+objid + " Month: " + month+" Day:"+day)
						print("total hours: " + str(total_time_hours))
						print("end hour: " + str(end_hour))
								
						
						r = end_hour - total_time_hours
						if r <= 0:
							if r == 0: x = 1
							else: x = 0
							for i in range(x, end_hour+1):
								hour_table_jobs_all_run[i] += 1
								print("Current day - 0..end hour:  " + str(i))
							
							#for negative numbers
							x = 24 - (total_time_hours - (end_hour+1))
							if x < 0: x = 0
								
							for i in range(x, 24):
								hour_table_jobs_all_run[i] += 1
								print("day before - start hour.. 23:  " + str(i))
						else:
							for i in range((end_hour - total_time_hours)+1, end_hour+1):
								hour_table_jobs_all_run[i] += 1
								print("same day - start hour.. 23:  " + str(i))
						print("___________________________________________________________")
							
						#sys.exit()	
						
						##############################################################################
						##############################################################################
						##############################################################################
						##############################################################################
					
				
						day_table[day_of_week] += 1
						month_table[month] += 1
						hour_table[hour] += 1
			except ValueError:
				print("//")
				
	
	
	#week day
	output_file_name = output_dir_name + '/' + 'workload_day_distribution_'+year+'.txt'
	output_file_txt = open(output_file_name, 'w')
	output_file_txt.write('DAY DAY_WEEK_JOBS\n')
	#print(day_table)
	output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), range(7), day_table.values())))
	output_file_txt.close()
	
	# month
	output_file_name = output_dir_name + '/' + 'workload_month_distribution_'+year+'.txt'
	output_file_txt = open(output_file_name, 'w')
	output_file_txt.write('MONTH MONTH_JOBS\n')
	output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), range(12), month_table.values())))
	output_file_txt.close()
	
	# hour
	output_file_name = output_dir_name + '/' + 'workload_hour_distribution_'+year+'.txt'
	output_file_txt = open(output_file_name, 'w')
	output_file_txt.write('HOUR HOUR_JOBS\n')
	output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), range(24), hour_table_jobs_all_run.values())))
	output_file_txt.close()
	

	# stop timer
	finishTime = time.clock()

	# printing summary
	print ("\nSUMMARY:                                          \n \
	%d files analyzed \n \
	%d jobs analyzed \n \
	%.3f seconds execution time" \
	% (file_count, job_count, finishTime-startTime))
	print("Total jobs year "+ year +": "+ str(job_total_count))
	return

### MAIN CODE ###
if len(sys.argv) >= 3:
	dir_name = sys.argv[1]
	output_dir_name = sys.argv[2]
	generate(dir_name, output_dir_name)
else:
	print ("ERROR, usage: %s <directory> <output directory>" % sys.argv[0])
	print ("<directory>: MOAB logs directory of a year")
	print ("<output directory>: directory to output frequency distributions")
	sys.exit(0)

