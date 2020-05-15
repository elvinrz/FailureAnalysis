
#!/usr/bin/env python

# University of Pittsburgh
# Center for Simulation and Modeling
# Esteban Meneses
# Date: 03/20/15

import sys
import re
import datetime
import time
import os
import glob
from math import *
import matplotlib as mtl
mtl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from collections import defaultdict
import collections
from datetime import date, timedelta

def init_tables(event_day):
	""" Initializes tables """
	
	event_day['00'] = 0
	event_day['01'] = 0
	event_day['02'] = 0
	event_day['03'] = 0
	event_day['04'] = 0
	event_day['05'] = 0
	event_day['06'] = 0
	event_day['07'] = 0
	event_day['08'] = 0
	event_day['09'] = 0
	event_day['10'] = 0
	event_day['11'] = 0
	event_day['12'] = 0
	event_day['13'] = 0
	event_day['14'] = 0
	event_day['15'] = 0
	event_day['16'] = 0
	event_day['17'] = 0
	event_day['18'] = 0
	event_day['19'] = 0
	event_day['20'] = 0
	event_day['21'] = 0
	event_day['22'] = 0
	event_day['23'] = 0
	
def full_dates(year1, year2, dictionary):
	dates_full_year = []
	d1 = date(int(year1), 1, 1)  	# start date
	d2 = date(int(year2), 12, 31)   # end date
	delta = d2 - d1         		# timedelta
	for i in range(delta.days + 1):
		r = str(d1 + timedelta(i))
		f = r[:4]+r[5:7]+r[8:10]
		print(f)
		dictionary[f] = 0
	#print(len(dictionary))
	return dictionary
	
def time_series(dir_name):
	#""" Reads a failure log file and correlates job IDs with MOAB log files in the directory """
	dayFormat = '%a_%b_%d_%Y'
	file_count = 0
	line_count = 0
	pathFileName = []
	
	event_day = {}
	event_week_2014 = {}
	event_week_2015 = {}
	event_week_2016 = {}
	event_week_2017 = {}
	event_week_2018 = {}
	
	
	event_week_2014_GPU_DBE  = {}
	event_week_2015_GPU_DBE  = {}
	event_week_2016_GPU_DBE  = {}
	event_week_2017_GPU_DBE  = {}
	event_week_2018_GPU_DBE  = {}
	
	event_week_2014_GPU_XID = {}
	event_week_2015_GPU_XID = {}
	event_week_2016_GPU_XID = {}
	event_week_2017_GPU_XID = {}
	event_week_2018_GPU_XID = {}
	
	#list init
	for i in range (1, 53):
		event_week_2014[i] = event_week_2014_GPU_DBE[i] = event_week_2014_GPU_XID[i] = 0
		event_week_2015[i] = event_week_2015_GPU_DBE[i] = event_week_2015_GPU_XID[i] = 0 
		event_week_2016[i] = event_week_2016_GPU_DBE[i] = event_week_2016_GPU_XID[i] = 0
		event_week_2017[i] = event_week_2017_GPU_DBE[i] = event_week_2017_GPU_XID[i] = 0
		event_week_2018[i] = event_week_2018_GPU_DBE[i] = event_week_2018_GPU_XID[i] = 0

	init = False
	# start timer
	startTime = time.clock()
    #get all files of the year
	for path, dirs, files in os.walk(dir_name):
			for f in files:
				pathFileName.append(f)
	
	
	init_tables(event_day)
		
	
	# going through all files in directory
	for file_name in pathFileName:  
		file_count += 1
		line_count = 0
		print("\nPrcessing %s "% file_name)
		count_event = 0
		file_name = dir_name + file_name
		
		with open(file_name) as log:
			line_count += 1
			if line_count == 1:
				next(log)
			for line in log:
				item = line.split("|")
				week = datetime.date(int(item[3][0:5]), int(item[3][6:8]), int(item[3][9:11])).isocalendar()[1]
				year = item[3][0:5].strip()
				category_error = item[6].strip()
				
				
				#initialize all key dates of a range to avoid null dates	
				# if init == False:
					# event_day = full_dates(2015,2016, event_day).copy()
					# init = True
				
				
				# if d in event_day.keys():
					# event_day[d] += 1
								
				if week in event_week_2014.keys():
					if year == "2014":		
						event_week_2014[week] += 1
						if category_error == "GPU DBE":
							event_week_2014_GPU_DBE[week] += 1
						if category_error == "GPU XID":
							event_week_2014_GPU_XID[week] += 1
						continue				
								
				if week in event_week_2015.keys():
					if year == "2015":		
						event_week_2015[week] += 1
						if category_error == "GPU DBE":
							event_week_2015_GPU_DBE[week] += 1
						if category_error == "GPU XID":
							event_week_2015_GPU_XID[week] += 1
						continue
				
				if week in event_week_2016.keys():
					if year == "2016":
						event_week_2016[week] += 1
						if category_error == "GPU DBE":
							event_week_2016_GPU_DBE[week] += 1
						if category_error == "GPU XID":
							event_week_2016_GPU_XID[week] += 1
						continue
						
				if week in event_week_2017.keys():
					if year == "2017":		
						event_week_2017[week] += 1
						if category_error == "GPU DBE":
							event_week_2017_GPU_DBE[week] += 1
						if category_error == "GPU XID":
							event_week_2017_GPU_XID[week] += 1
						continue		

				if week in event_week_2018.keys():
					if year == "2018":		
						event_week_2018[week] += 1
						if category_error == "GPU DBE":
							event_week_2018_GPU_DBE[week] += 1
						if category_error == "GPU XID":
							event_week_2018_GPU_XID[week] += 1
						continue				
										
				
				
	# print("\nPrcessing %d year of 2 - Processing 1 plots of 3"% file_count)	
	# plt.style.use('seaborn-whitegrid')	
	# plt.xlabel('Days')
	# plt.ylabel('Count of failures')
	# plt.title('Failures by day 2015-2016 ')
	# ax = plt.gca()
	# ax.tick_params(axis = 'x', which = 'major', labelsize = 6)
	# day_sort = collections.OrderedDict(sorted(event_day.items()))
	# plt.xticks(np.arange(0, 730, 30))
	# plt.figure(figsize=(12,4)) 
	# plt.plot(range(len(event_day)),list(day_sort.values()), 'b-', linewidth=1, label='2015-2016 failures')
	# plt.legend(framealpha=1,shadow=True, borderpad = 1, fancybox=True)
		
	# plt.savefig("PLOT_day_2015_2016.pdf")
	# print("\nPlot in file: <PLOT_day_2015_2016.pdf>")

	print("Total 2014: "+ str(event_week_2014.values()))
	print("Total 2015: "+ str(event_week_2015.values()))
	print("Total 2016: "+ str(event_week_2016.values()))
	print("Total 2017: "+ str(event_week_2017.values()))
	print("Total 2018: "+ str(event_week_2018.values()))
 
	print("\nPrcessing 1 plot of 2")	
	plt.clf()
	#plt.style.use('seaborn-whitegrid')	
	plt.xlabel('Weeks')
	plt.ylabel('Count of failures')
	plt.title('Failures by week 2014-2018 ')
	ax = plt.gca()
	ax.tick_params(axis = 'x', which = 'major', labelsize = 6)
	plt.xticks(np.arange(0, 54, 2))
	plt.plot(range(len(event_week_2014)),list(event_week_2014.values()), 'g--', linewidth=0.5, label="2014 failures")
	plt.plot(range(len(event_week_2015)),list(event_week_2015.values()), 'g-', linewidth=0.5, label="2015 failures")
	plt.plot(range(len(event_week_2016)),list(event_week_2016.values()), 'r-', linewidth=1, label="2016 failures")
	plt.plot(range(len(event_week_2017)),list(event_week_2017.values()), 'm-', linewidth=0.5, label="2017 failures")
	plt.plot(range(len(event_week_2018)),list(event_week_2018.values()), 'c-', linewidth=0.5, label="2018 failures")
	
	plt.legend(framealpha=1,shadow=True, borderpad = 1, fancybox=True)
	plt.savefig("PLOT_week_2014_2018.pdf")
	print("\nPlot in file: <PLOT_week_2014_2018.pdf>")

	
	print("\nProcessing 2 plot of 2")	
	
	plt.clf()
	plt.figure(figsize=(8,3))
	plt.xlabel('Weeks')
	ax = plt.gca()
	ax.tick_params(axis = 'x', which = 'major', labelsize = 6)
	plt.ylabel('Failure Count')
	plt.title('Failures by week')
	plt.xticks(np.arange(0, 264, 8))
	plt.xticks(rotation=45)
	plt.plot(range(len(event_week_2014)+len(event_week_2015)+len(event_week_2016)+len(event_week_2017)+len(event_week_2018)),list(event_week_2014.values())+list(event_week_2015.values())+list(event_week_2016.values())+list(event_week_2017.values())+list(event_week_2018.values()), 'b-', linewidth=0.7, label="Total Failures")
	
	plt.plot(range(len(event_week_2014_GPU_DBE)+len(event_week_2015_GPU_DBE)+len(event_week_2016_GPU_DBE)+len(event_week_2017_GPU_DBE)+len(event_week_2018_GPU_DBE)),list(event_week_2014_GPU_DBE.values())+list(event_week_2015_GPU_DBE.values())+list(event_week_2016_GPU_DBE.values())+list(event_week_2017_GPU_DBE.values())+list(event_week_2018_GPU_DBE.values()), 'r-', linewidth=0.4, label="GPU DBE Failures")
	
	plt.plot(range(len(event_week_2014_GPU_XID)+len(event_week_2015_GPU_XID)+len(event_week_2016_GPU_XID)+len(event_week_2017_GPU_XID)+len(event_week_2018_GPU_XID)),list(event_week_2014_GPU_XID.values())+list(event_week_2015_GPU_XID.values())+list(event_week_2016_GPU_XID.values())+list(event_week_2017_GPU_XID.values())+list(event_week_2018_GPU_XID.values()), 'g-', linewidth=0.4, label="GPU XID Failures")
	
	
	#plt.legend(framealpha=0.5,shadow=False, borderpad = 1, fancybox=False,prop={'size': 6})
	plt.axvline(x=52, color='k', linestyle='dashed', linewidth=0.7)
	plt.axvline(x=104, color='k', linestyle='dashed', linewidth=0.7)
	plt.axvline(x=156, color='k', linestyle='dashed', linewidth=0.7)
	plt.axvline(x=208, color='k', linestyle='dashed', linewidth=0.7)
	
	plt.text(10 ,130,"2014")
	plt.text(68 ,130,"2015")
	plt.text(120 ,130,"2016")
	plt.text(173 ,130,"2017")
	plt.text(212 ,130,"2018")
	
	plt.legend(edgecolor="black",prop={'size': 6},loc=0)
	
	plt.tight_layout()
	plt.savefig("PLOT_all_years_week.pdf")
	print("\nPlot in file: <PLOT_all_years_week.pdf>")
	
	return 
	
	
if len(sys.argv) >= 2:
	dirName = sys.argv[1]
	outputFileName = "" #sys.argv[2]
	time_series(dirName)
else:
	print ("ERROR, usage: %s <directory> <output file>" % sys.argv[0])
	sys.exit(0)