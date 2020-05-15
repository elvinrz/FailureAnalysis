#!/usr/bin/env python

# University of Pittsburgh
# Center for Simulation and Modeling
# Esteban Meneses
# Extracts information about jobs from MOAB logs.
# Date: 04/04/15

import sys
import re
import datetime
import time
import os
import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from math import *
import calendar as cl
import datetime

from datetime import date, timedelta, datetime as dt

### FUNCTIONS ###


def init_tables(jobs_time_execution, event_day_year, event_week_daymonth,event_week_year, hour_table, day_table, month_table):
	""" Initializes tables """
	
	for i in range(19):
		jobs_time_execution[i] = 0
		
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
	
	for i in range(365):
		if i not in event_day_year.keys():
			event_day_year[i] = 0
	
	for i in range(1,32):
		event_week_daymonth[i] = 0
	
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
	day_table['Monday'] = 0
	day_table['Tuesday'] = 0
	day_table['Wednesday'] = 0
	day_table['Thursday'] = 0
	day_table['Friday'] = 0
	day_table['Saturday'] = 0
	day_table['Sunday'] = 0
	month_table['01'] = 0
	month_table['02'] = 0
	month_table['03'] = 0
	month_table['04'] = 0
	month_table['05'] = 0
	month_table['06'] = 0
	month_table['07'] = 0
	month_table['08'] = 0
	month_table['09'] = 0
	month_table['10'] = 0
	month_table['11'] = 0
	month_table['12'] = 0

	for i in range (1, 53):
		event_week_year[i] = 0

def read_files(*filenames):
    for filename in filenames:
        with open(filename,'r') as file_obj:
            for line in file_obj:
                yield line

def generate(dir_name, output_dir_name):
	#""" Reads a failure log file and correlates job IDs with MOAB log files in the directory """
	format = '%Y-%m-%d %H:%M:%S'
	file_count = 0
	job_count = 0
	count_ommited = 0
	job_total_count = 0
	hour_table = {}
	day_table = {}
	month_table = {}
	pathFileName = []
	event_week_year = {}
	encontrado = 0
	event_week_daymonth = {}
	event_day_year = {}
	jobs_time_execution = {}
	jobs_time_list = []
	cancelled_jobs = 0
	
	
	init_tables(jobs_time_execution, event_day_year, event_week_daymonth, event_week_year, hour_table,day_table,month_table)

	
	#get all files of the year
	for path, dirs, files in os.walk(dir_name):
		for d in dirs:
			for f in glob.iglob(os.path.join(path, d, '*')):
				pathFileName.append(f)
	
	#pathFileName.sort()
	
	sizeList = len(pathFileName)
	# going through all files in directory
	for file_name in pathFileName:
		file_count = file_count + 1		
		# getting day and month
		day = file_name[-2:]
		month = file_name[-5:-3]
		year = file_name[-10:-6]
		
		print ("Progress: %d%%, Month Analized: %s, jobs: %d, found: %d"% (file_count/sizeList*100, month, job_count, encontrado),end="\r") 
		sys.stdout.flush()
			
		# l.clear()
		# index = pathFileName.index(file_name)
		# if index < len(pathFileName)-4:
			# l.append(pathFileName[index])
			# l.append(pathFileName[index + 1])
			# l.append(pathFileName[index + 2])
			# l.append(pathFileName[index + 3])
		
		for event in read_files(file_name):		
			
			columns = event.split()
			eventType = columns[2]
			objid = columns[3]
			objEvent = columns[4]
			
			if len(columns) < 6 or eventType != 'job':
				continue														# continue if empty event
			
			# checking for a valid job entry
			if len(columns) > 3 and objEvent == 'JOBEND':
				job_count = job_count + 1
				job_time = columns[0]
				hour = job_time.split(':')[0]
				hour_table[hour] = hour_table[hour] + 1
				
				
				columns_check = event.split()
				if "STARTTIME" in event:
					for item in columns_check:			
						if "STARTTIME" in item:
							#start_time
							starttime = item[10:]		#STARTTIME	
						if "COMPLETETIME" in item:	
							#completion_time
							completetime = item[13:]		#COMPLETETIME
				else:
					starttime = columns[14]
					completetime = columns[15]
				try:	
					execution_time = (int(completetime) - int(starttime)) / 60
				except ValueError:
					print("Value error.  Value ommited: "+ str(count_ommited))
					count_ommited += 1
				
				jobs_time_list.append(objid +" hour start:"+hour+" Execution Time:"+str(execution_time))
				if int(execution_time) in jobs_time_execution:
					jobs_time_execution[int(execution_time)] += 1
				else:
					if int(execution_time) > 19 and int(execution_time) < 30:
						jobs_time_execution[20] += 1
					if int(execution_time) > 29 and int(execution_time) < 40:
						jobs_time_execution[30] += 1
					if int(execution_time) > 39 and int(execution_time) < 50:
						jobs_time_execution[40] += 1
					if int(execution_time) > 49 and int(execution_time) < 100:
						jobs_time_execution[50] += 1
					if int(execution_time) > 99 and int(execution_time) < 150:
						jobs_time_execution[100] += 1
					if int(execution_time) > 149 and int(execution_time) < 200:
						jobs_time_execution[150] += 1	
					if int(execution_time) > 199 and int(execution_time) < 250:
						jobs_time_execution[200] += 1
					if int(execution_time) > 249 and int(execution_time) < 300:
						jobs_time_execution[250] += 1
					if int(execution_time) > 299 and int(execution_time) < 350:
						jobs_time_execution[300] += 1
					if int(execution_time) > 349 and int(execution_time) < 400:
						jobs_time_execution[350] += 1
					if int(execution_time) > 399 and int(execution_time) < 450:
						jobs_time_execution[400] += 1
					if int(execution_time) > 449 and int(execution_time) < 500:
						jobs_time_execution[450] += 1
					if int(execution_time) > 499 and int(execution_time) < 550:
						jobs_time_execution[500] += 1
					if int(execution_time) > 549 and int(execution_time) < 600:
						jobs_time_execution[550] += 1
					if int(execution_time) >= 600:
						jobs_time_execution[600] += 1
							
	# # for save data of day year
	# output_file_name = output_dir_name + '/' + 'workload_day_year_distribution_'+year+'.txt'
	# output_file_txt = open(output_file_name, 'w')
	# output_file_txt.write('DAY DAY_YEAR_JOBS\n')
	# #print(event_week_daymonth)
	# output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), range(365), event_day_year.values())))
	# output_file_txt.close()
	
	# # creating output file
	print ("Generating workload graphic by hours") 
	
	data = []
	for key in sorted(jobs_time_execution.keys()):	
		data.append(jobs_time_execution[key])
		
	print(data)
	
	# #plt.style.use('seaborn-whitegrid')
	fig = plt.figure()
	fig, axs = plt.subplots(2, 2,figsize=(8, 5))
	
	
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
	# ################################################################################################################
	
	#save all the plots
	plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95, hspace=0.35, wspace=0.25)	
	plt.savefig("PLOT_Jobs_times_"+year+".pdf")
	

	
	# for c in jobs_time_list:
		# print(c)

	# print(jobs_time_execution)	
	# # printing summary
	print ("\nSUMMARY:                                          \n \
	%d files analyzed \n \
	%d jobs analyzed " 
	% (file_count, job_count))
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

