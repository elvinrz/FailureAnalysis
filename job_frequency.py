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

### FUNCTIONS ###

def init_tables(hour_table, day_table, month_table):
	""" Initializes tables """
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
	day_table['01'] = 0
	day_table['02'] = 0
	day_table['03'] = 0
	day_table['04'] = 0
	day_table['05'] = 0
	day_table['06'] = 0
	day_table['07'] = 0
	day_table['08'] = 0
	day_table['09'] = 0
	day_table['10'] = 0
	day_table['11'] = 0
	day_table['12'] = 0
	day_table['13'] = 0
	day_table['14'] = 0
	day_table['15'] = 0
	day_table['16'] = 0
	day_table['17'] = 0
	day_table['18'] = 0
	day_table['19'] = 0
	day_table['20'] = 0
	day_table['21'] = 0
	day_table['22'] = 0
	day_table['23'] = 0
	day_table['24'] = 0
	day_table['25'] = 0
	day_table['26'] = 0
	day_table['27'] = 0
	day_table['28'] = 0
	day_table['29'] = 0
	day_table['30'] = 0
	day_table['31'] = 0
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

def generate(dir_name, output_dir_name):
	""" Reads a failure log file and correlates job IDs with MOAB log files in the directory """
	file_count = 0
	job_count = 0
	hour_table = {}
	day_table = {}
	month_table = {}
	pathFileName = []
	
	init_tables(hour_table,day_table,month_table)

	# start timer
	startTime = time.clock()
	
	#get all files of the year
	for path, dirs, files in os.walk(dir_name):
		for d in dirs:
			for f in glob.iglob(os.path.join(path, d, '*')):
				pathFileName.append(f)
	
	sizeList = len(pathFileName)
	# going through all files in directory
	for file_name in pathFileName:
		file_count = file_count + 1

		# getting day and month
		day = file_name[-2:]
		month = file_name[-5:-3]
		
		print ("Progress: %d%%, Month Analized: %s"% (file_count/sizeList*100, month),end="\r") 
		sys.stdout.flush()
				
		with open(file_name) as log:
			for event in log:
				columns = event.split()
				if len(columns) < 6:
					continue														# continue if empty event
				eventType = columns[2]
				if eventType != 'job':
					continue														# continue if not job event
				objid = columns[3]
				objEvent = columns[4]

				# checking for a valid job entry
				if len(columns) > 3 and objEvent == 'JOBSUBMIT':
					job_count = job_count + 1
					job_time = columns[0]
					hour = job_time.split(':')[0]
					#print ("->%s %s %s<-" % (hour,day,month))
					hour_table[hour] = hour_table[hour] + 1
					day_table[day] = day_table[day] + 1
					month_table[month] = month_table[month] + 1

	print ("\r                                                           ",) 
	
	# creating output file
	print ("Generating graphics 1 of 3") 
		
	output_file_name = output_dir_name + '/' + 'workload_hour_distribution.pdf'
	data = [hour_table['00'],hour_table['01'],hour_table['02'],hour_table['03'],hour_table['04'],hour_table['05'],hour_table['06'],hour_table['07'],hour_table['08'],hour_table['09'],hour_table['10'],hour_table['11'],hour_table['12'],hour_table['13'],hour_table['14'],hour_table['15'],hour_table['16'],hour_table['17'],hour_table['18'],hour_table['19'],hour_table['20'],hour_table['21'],hour_table['22'],hour_table['23'],]
	plt.clf()
	plt.rc('font', family='DejaVu Sans')
	plt.rc('font', serif='Times New Roman')
	plt.rc('font', size=24)
	index = np.arange(24)
	plt.bar(index, data)
	plt.xlabel('Hour of the Day')
	plt.xticks([2.4,6.4,10.4,14.4,18.4,22.4], ['2','6','10','14','18','22'])
	plt.ylabel('Number of Jobs')
	plt.savefig(output_file_name,bbox_inches='tight')
	output_file_name = output_dir_name + '/' + 'workload_hour_distribution.txt'
	output_file_txt = open(output_file_name, 'w')
	output_file_txt.write('#HOUR JOBS\n')
	output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), range(24), data)))
	output_file_txt.close()
	
	print ("Generating graphics 2 of 3")
	output_file_name = output_dir_name + '/' + 'workload_day_distribution.pdf'
	#ORIGINAL data = [day_table['Sun'],day_table['Mon'],day_table['Tue'],day_table['Wed'],day_table['Thu'],day_table['Fri'],day_table['Sat']]
	data = [day_table['01'],day_table['02'],day_table['03'],day_table['04'],day_table['05'],day_table['06'],day_table['07'],day_table['08'],day_table['09'],day_table['10'],day_table['11'],day_table['12'],day_table['13'],day_table['14'],day_table['15'],day_table['16'],day_table['17'],day_table['18'],day_table['19'],day_table['20'],day_table['21'],day_table['22'],day_table['23'],day_table['24'],day_table['25'],day_table['26'],day_table['27'],day_table['28'],day_table['29'],day_table['30'],day_table['31']]
	plt.clf()
	plt.rc('font', family='DejaVu Sans')
	plt.rc('font', serif='Times New Roman')
	plt.rc('font', size=7)
	#ORIGINAL plt.bar(range(7), data)
	plt.bar(range(31), data)
	#ORIGINAL plt.xticks([0.4,1.4,2.4,3.4,4.4,5.4,6.4], ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'])
	plt.xticks([0.1,1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1,10.1,11.1,12.1,13.1,14.1,15.1,16.1,17.1,18.1,19.1,20.1,21.1,22.1,23.1,24.1,25.1,26.1,27.1,28.1,29.1,30.1,31.1], ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'])
	plt.xlabel('Day of the Month')
	plt.ylabel('Number of Jobs')
	plt.savefig(output_file_name,bbox_inches='tight')
	output_file_name = output_dir_name + '/' + 'workload_day_distribution.txt'
	output_file_txt = open(output_file_name, 'w')
	output_file_txt.write('#DAY JOBS\n')
	output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), range(7), data)))
	output_file_txt.close()
	
	print ("Generating graphics 3 of 3")
	output_file_name = output_dir_name + '/' + 'workload_month_distribution.pdf'
	#ORIGINAL data = [month_table['Jan'],month_table['Feb'],month_table['Mar'],month_table['Apr'],month_table['May'],month_table['Jun'],month_table['Jul'],month_table['Aug'],month_table['Sep'],month_table['Oct'],month_table['Nov'],month_table['Dec']]
	data = [month_table['01'],month_table['02'],month_table['03'],month_table['04'],month_table['05'],month_table['06'],month_table['07'],month_table['08'],month_table['09'],month_table['10'],month_table['11'],month_table['12']]
	plt.clf()
	plt.rc('font', family='DejaVu Sans')
	plt.rc('font', serif='Times New Roman')
	plt.rc('font', size=24)
	plt.bar(range(12), data)
	plt.xticks([0.4,1.4,2.4,3.4,4.4,5.4,6.4,7.4,8.4,9.4,10.4,11.4], ['1','2','3','4','5','6','7','8','9','10','11','12'])
	plt.xlabel('Month of the Year')
	plt.ylabel('Number of Jobs')
	plt.savefig(output_file_name,bbox_inches='tight')
	output_file_name = output_dir_name + '/' + 'workload_month_distribution.txt'
	output_file_txt = open(output_file_name, 'w')
	output_file_txt.write('#MONTH JOBS\n')
	output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), range(12), data)))
	output_file_txt.close()
	
	# stop timer
	finishTime = time.clock()

	# printing summary
	print ("\nSUMMARY:                                          \n \
	%d files analyzed \n \
	%d jobs analyzed \n \
	%.3f seconds execution time" \
	% (file_count, job_count, finishTime-startTime))
	
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

