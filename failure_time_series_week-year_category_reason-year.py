
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

def init_tables_category_reason(event_category,event_category_2015,event_category_2016,event_reason, event_reason_2015, event_reason_2016):
	event_category['hardware'] = 0
	event_category['software'] = 0
	event_reason['user'] = 0
	event_reason['system'] = 0
	
	event_category_2015['hardware'] = 0
	event_category_2015['software'] = 0
	event_reason_2015['user'] = 0
	event_reason_2015['system'] = 0

	event_category_2016['hardware'] = 0
	event_category_2016['software'] = 0
	event_reason_2016['user'] = 0
	event_reason_2016['system'] = 0

def init_tables(event_day_hard,event_day_soft,event_week_hard_2015,event_week_soft_2015,event_week_hard_2016,event_week_soft_2016):
	""" Initializes tables """
	event_day_hard['00'] = 0
	event_day_hard['01'] = 0
	event_day_hard['02'] = 0
	event_day_hard['03'] = 0
	event_day_hard['04'] = 0
	event_day_hard['05'] = 0
	event_day_hard['06'] = 0
	event_day_hard['07'] = 0
	event_day_hard['08'] = 0
	event_day_hard['09'] = 0
	event_day_hard['10'] = 0
	event_day_hard['11'] = 0
	event_day_hard['12'] = 0
	event_day_hard['13'] = 0
	event_day_hard['14'] = 0
	event_day_hard['15'] = 0
	event_day_hard['16'] = 0
	event_day_hard['17'] = 0
	event_day_hard['18'] = 0
	event_day_hard['19'] = 0
	event_day_hard['20'] = 0
	event_day_hard['21'] = 0
	event_day_hard['22'] = 0
	event_day_hard['23'] = 0
	event_day_soft['00'] = 0
	event_day_soft['01'] = 0
	event_day_soft['02'] = 0
	event_day_soft['03'] = 0
	event_day_soft['04'] = 0
	event_day_soft['05'] = 0
	event_day_soft['06'] = 0
	event_day_soft['07'] = 0
	event_day_soft['08'] = 0
	event_day_soft['09'] = 0
	event_day_soft['10'] = 0
	event_day_soft['11'] = 0
	event_day_soft['12'] = 0
	event_day_soft['13'] = 0
	event_day_soft['14'] = 0
	event_day_soft['15'] = 0
	event_day_soft['16'] = 0
	event_day_soft['17'] = 0
	event_day_soft['18'] = 0
	event_day_soft['19'] = 0
	event_day_soft['20'] = 0
	event_day_soft['21'] = 0
	event_day_soft['22'] = 0
	event_day_soft['23'] = 0

	event_week_hard_2015[1] = 0
	event_week_hard_2015[2] = 0
	event_week_hard_2015[3] = 0
	event_week_hard_2015[4] = 0
	event_week_hard_2015[5] = 0
	event_week_hard_2015[6] = 0
	event_week_hard_2015[7] = 0
	event_week_hard_2015[8] = 0
	event_week_hard_2015[9] = 0
	event_week_hard_2015[10] = 0
	event_week_hard_2015[11] = 0
	event_week_hard_2015[12] = 0
	event_week_hard_2015[13] = 0
	event_week_hard_2015[14] = 0
	event_week_hard_2015[15] = 0
	event_week_hard_2015[16] = 0
	event_week_hard_2015[17] = 0
	event_week_hard_2015[18] = 0
	event_week_hard_2015[19] = 0
	event_week_hard_2015[20] = 0
	event_week_hard_2015[21] = 0
	event_week_hard_2015[22] = 0
	event_week_hard_2015[23] = 0
	event_week_hard_2015[24] = 0
	event_week_hard_2015[25] = 0
	event_week_hard_2015[26] = 0
	event_week_hard_2015[27] = 0
	event_week_hard_2015[28] = 0
	event_week_hard_2015[29] = 0
	event_week_hard_2015[30] = 0
	event_week_hard_2015[31] = 0
	event_week_hard_2015[32] = 0
	event_week_hard_2015[33] = 0
	event_week_hard_2015[34] = 0
	event_week_hard_2015[35] = 0
	event_week_hard_2015[36] = 0
	event_week_hard_2015[37] = 0
	event_week_hard_2015[38] = 0
	event_week_hard_2015[39] = 0
	event_week_hard_2015[40] = 0
	event_week_hard_2015[41] = 0
	event_week_hard_2015[42] = 0
	event_week_hard_2015[43] = 0
	event_week_hard_2015[44] = 0
	event_week_hard_2015[45] = 0
	event_week_hard_2015[46] = 0
	event_week_hard_2015[47] = 0
	event_week_hard_2015[48] = 0
	event_week_hard_2015[49] = 0
	event_week_hard_2015[50] = 0
	event_week_hard_2015[51] = 0
	event_week_hard_2015[52] = 0
	event_week_hard_2016[1] = 0
	event_week_hard_2016[2] = 0
	event_week_hard_2016[3] = 0
	event_week_hard_2016[4] = 0
	event_week_hard_2016[5] = 0
	event_week_hard_2016[6] = 0
	event_week_hard_2016[7] = 0
	event_week_hard_2016[8] = 0
	event_week_hard_2016[9] = 0
	event_week_hard_2016[10] = 0
	event_week_hard_2016[11] = 0
	event_week_hard_2016[12] = 0
	event_week_hard_2016[13] = 0
	event_week_hard_2016[14] = 0
	event_week_hard_2016[15] = 0
	event_week_hard_2016[16] = 0
	event_week_hard_2016[17] = 0
	event_week_hard_2016[18] = 0
	event_week_hard_2016[19] = 0
	event_week_hard_2016[20] = 0
	event_week_hard_2016[21] = 0
	event_week_hard_2016[22] = 0
	event_week_hard_2016[23] = 0
	event_week_hard_2016[24] = 0
	event_week_hard_2016[25] = 0
	event_week_hard_2016[26] = 0
	event_week_hard_2016[27] = 0
	event_week_hard_2016[28] = 0
	event_week_hard_2016[29] = 0
	event_week_hard_2016[30] = 0
	event_week_hard_2016[31] = 0
	event_week_hard_2016[32] = 0
	event_week_hard_2016[33] = 0
	event_week_hard_2016[34] = 0
	event_week_hard_2016[35] = 0
	event_week_hard_2016[36] = 0
	event_week_hard_2016[37] = 0
	event_week_hard_2016[38] = 0
	event_week_hard_2016[39] = 0
	event_week_hard_2016[40] = 0
	event_week_hard_2016[41] = 0
	event_week_hard_2016[42] = 0
	event_week_hard_2016[43] = 0
	event_week_hard_2016[44] = 0
	event_week_hard_2016[45] = 0
	event_week_hard_2016[46] = 0
	event_week_hard_2016[47] = 0
	event_week_hard_2016[48] = 0
	event_week_hard_2016[49] = 0
	event_week_hard_2016[50] = 0
	event_week_hard_2016[51] = 0
	event_week_hard_2016[52] = 0
	event_week_soft_2015[1] = 0
	event_week_soft_2015[2] = 0
	event_week_soft_2015[3] = 0
	event_week_soft_2015[4] = 0
	event_week_soft_2015[5] = 0
	event_week_soft_2015[6] = 0
	event_week_soft_2015[7] = 0
	event_week_soft_2015[8] = 0
	event_week_soft_2015[9] = 0
	event_week_soft_2015[10] = 0
	event_week_soft_2015[11] = 0
	event_week_soft_2015[12] = 0
	event_week_soft_2015[13] = 0
	event_week_soft_2015[14] = 0
	event_week_soft_2015[15] = 0
	event_week_soft_2015[16] = 0
	event_week_soft_2015[17] = 0
	event_week_soft_2015[18] = 0
	event_week_soft_2015[19] = 0
	event_week_soft_2015[20] = 0
	event_week_soft_2015[21] = 0
	event_week_soft_2015[22] = 0
	event_week_soft_2015[23] = 0
	event_week_soft_2015[24] = 0
	event_week_soft_2015[25] = 0
	event_week_soft_2015[26] = 0
	event_week_soft_2015[27] = 0
	event_week_soft_2015[28] = 0
	event_week_soft_2015[29] = 0
	event_week_soft_2015[30] = 0
	event_week_soft_2015[31] = 0
	event_week_soft_2015[32] = 0
	event_week_soft_2015[33] = 0
	event_week_soft_2015[34] = 0
	event_week_soft_2015[35] = 0
	event_week_soft_2015[36] = 0
	event_week_soft_2015[37] = 0
	event_week_soft_2015[38] = 0
	event_week_soft_2015[39] = 0
	event_week_soft_2015[40] = 0
	event_week_soft_2015[41] = 0
	event_week_soft_2015[42] = 0
	event_week_soft_2015[43] = 0
	event_week_soft_2015[44] = 0
	event_week_soft_2015[45] = 0
	event_week_soft_2015[46] = 0
	event_week_soft_2015[47] = 0
	event_week_soft_2015[48] = 0
	event_week_soft_2015[49] = 0
	event_week_soft_2015[50] = 0
	event_week_soft_2015[51] = 0
	event_week_soft_2015[52] = 0
	event_week_soft_2016[1] = 0
	event_week_soft_2016[2] = 0
	event_week_soft_2016[3] = 0
	event_week_soft_2016[4] = 0
	event_week_soft_2016[5] = 0
	event_week_soft_2016[6] = 0
	event_week_soft_2016[7] = 0
	event_week_soft_2016[8] = 0
	event_week_soft_2016[9] = 0
	event_week_soft_2016[10] = 0
	event_week_soft_2016[11] = 0
	event_week_soft_2016[12] = 0
	event_week_soft_2016[13] = 0
	event_week_soft_2016[14] = 0
	event_week_soft_2016[15] = 0
	event_week_soft_2016[16] = 0
	event_week_soft_2016[17] = 0
	event_week_soft_2016[18] = 0
	event_week_soft_2016[19] = 0
	event_week_soft_2016[20] = 0
	event_week_soft_2016[21] = 0
	event_week_soft_2016[22] = 0
	event_week_soft_2016[23] = 0
	event_week_soft_2016[24] = 0
	event_week_soft_2016[25] = 0
	event_week_soft_2016[26] = 0
	event_week_soft_2016[27] = 0
	event_week_soft_2016[28] = 0
	event_week_soft_2016[29] = 0
	event_week_soft_2016[30] = 0
	event_week_soft_2016[31] = 0
	event_week_soft_2016[32] = 0
	event_week_soft_2016[33] = 0
	event_week_soft_2016[34] = 0
	event_week_soft_2016[35] = 0
	event_week_soft_2016[36] = 0
	event_week_soft_2016[37] = 0
	event_week_soft_2016[38] = 0
	event_week_soft_2016[39] = 0
	event_week_soft_2016[40] = 0
	event_week_soft_2016[41] = 0
	event_week_soft_2016[42] = 0
	event_week_soft_2016[43] = 0
	event_week_soft_2016[44] = 0
	event_week_soft_2016[45] = 0
	event_week_soft_2016[46] = 0
	event_week_soft_2016[47] = 0
	event_week_soft_2016[48] = 0
	event_week_soft_2016[49] = 0
	event_week_soft_2016[50] = 0
	event_week_soft_2016[51] = 0
	event_week_soft_2016[52] = 0

def time_series(dir_name, outputFileName):
	#""" Reads a failure log file and correlates job IDs with MOAB log files in the directory """
	dayFormat = '%a_%b_%d_%Y'
	file_count = 0
	line_count = 0
	pathFileName = []
	event_day_hard = {}
	event_day_soft = {}
	event_week_hard_2015 = {}
	event_week_soft_2015 = {}
	event_week_hard_2016 = {}
	event_week_soft_2016 = {}
	
	event_category = {}
	event_reason = {}
	
	event_category_2015 = {}
	event_reason_2015 = {}
	
	event_category_2016 = {}
	event_reason_2016 = {}
	
	# start timer
	startTime = time.clock()
    #get all files of the year
	for path, dirs, files in os.walk(dir_name):
			for f in files:
				pathFileName.append(f)
				
	init_tables(event_day_hard,event_day_soft,event_week_hard_2015,event_week_soft_2015,event_week_hard_2016,event_week_soft_2016)
	init_tables_category_reason(event_category,event_category_2015,event_category_2016,event_reason, event_reason_2015, event_reason_2016)

	# going through all files in directory
	for file_name in pathFileName:  
		file_count += 1
		line_count = 0
		print("\nPrcessing year %d"% file_count)
		file_name = dir_name + file_name
		
		with open(file_name) as log:
			line_count += 1
			if line_count == 1:
				next(log)
			for line in log:
				item = line.split("|")
				
				d = item[3].strip()[5:-12]#item[3].strip()[:-15]+item[3].strip()[5:-12]+item[3].strip()[8:-9]
				week = datetime.date(int(item[3].strip()[:-15]), int(item[3].strip()[5:-12]), int(item[3].strip()[8:-9])).isocalendar()[1]
				year = item[3].strip()[:-15]
				category = item[4].strip()
				reason = item[5].strip()
				
				#############################################################
				#count category and reason events
				event_category[category] += 1
				event_reason[reason] += 1
				if year == "2015":
					event_category_2015[category] += 1
					event_reason_2015[reason] += 1
				else:
					if year == "2016":
						event_category_2016[category] += 1
						event_reason_2016[reason] += 1
							
				#############################################################
				#############################################################
				
				#############################################################
				#DAY
				if d in event_day_hard.keys():
					if category == "hardware":
						event_day_hard[d] += 1
				
				if d in event_day_soft.keys():
					if category == "software":
						event_day_soft[d] += 1

					
				#############################################################
				#WEEK HARDWARE YEAR
				if week in event_week_hard_2015.keys():
					if year == "2015":
						if category == "hardware":
							event_week_hard_2015[week] += 1
						if category == "software":
							event_week_soft_2015[week] += 1
						continue
					
				#WEEK SOFTWARE YEAR
				if week in event_week_soft_2016.keys():
					if year == "2016":
						if category == "hardware":
							event_week_hard_2016[week] += 1
						if category == "software":
							event_week_soft_2016[week] += 1
	
	# print("\nPrcessing %d year of 2 - Processing 1 plots of 4"% file_count)			
	# plt.style.use('seaborn-whitegrid')	
	# plt.xlabel('Days')
	# plt.ylabel('Count of failures')
	# plt.title('Failures by day 2015-2016 ')
	# ax = plt.gca()
	# ax.tick_params(axis = 'x', which = 'major', labelsize = 6)
	
	# day_soft_sort = collections.OrderedDict(sorted(event_day_soft.items()))
	# day_hard_sort = collections.OrderedDict(sorted(event_day_hard.items()))
	# plt.plot(range(len(event_day_soft)),list(day_soft_sort.values()), 'b-', linewidth=1, label='2015-2016 software failures ')
	# plt.plot(range(len(event_day_hard)),list(day_hard_sort.values()), 'r-', linewidth=1, label='2015-2016 hardware failures')
	# plt.legend(framealpha=1,shadow=True, borderpad = 1, fancybox=True)
	# plt.savefig("PLOT_day_allyear_category.pdf")
	# print("\nPlot in file: <PLOT_day_allyear_category.pdf>")
	
	plt.style.use('seaborn-whitegrid')
	fig = plt.figure()
	fig, axs = plt.subplots(2,3,figsize=(15, 8))
	
	print("\nProcessing plots 1")
		
	plt.style.use('seaborn-whitegrid')	
	axs[0,0].set_xlabel('Weeks')
	axs[0,0].set_ylabel('Count of failures')
	axs[0,0].set_title('Failures by week 2015')
	ax = plt.gca()
	axs[0,0].tick_params(axis = 'x', which = 'major', labelsize = 6)
	axs[0,0].set_xticks(np.arange(0, 54, 2))
	axs[0,0].plot(range(len(event_week_soft_2015)),list(event_week_soft_2015.values()), 'b-', linewidth=1, label="2015 Software failures")
	axs[0,0].plot(range(len(event_week_hard_2015)),list(event_week_hard_2015.values()), 'r--', linewidth=1, label="2015 Hardware failures")
	axs[0,0].legend(framealpha=1,shadow=True, borderpad = 1, fancybox=True)
	
	print("\nProcessing plots 2")
	plt.style.use('seaborn-whitegrid')	
	axs[0,1].set_xlabel('Weeks')
	axs[0,1].set_ylabel('Count of failures')
	axs[0,1].set_title('Failures by week 2016 ')
	ax = plt.gca()
	axs[0,1].tick_params(axis = 'x', which = 'major', labelsize = 6)
	axs[0,1].set_xticks(np.arange(0, 54, 4))
	axs[0,1].plot(range(len(event_week_soft_2016)),list(event_week_soft_2016.values()), 'b-', linewidth=1, label="2016 Software failures")
	axs[0,1].plot(range(len(event_week_hard_2016)),list(event_week_hard_2016.values()), 'r--', linewidth=1, label="2016 Hardware failures")
	axs[0,1].legend(framealpha=1,shadow=True, borderpad = 1, fancybox=True)
	
	print("\nProcessing plots 3")
	axs[0,2].set_xlabel('Weeks')
	ax = plt.gca()
	axs[0,2].tick_params(axis = 'x', which = 'major', labelsize = 6)
	axs[0,2].set_ylabel('Count of failures')
	axs[0,2].set_title('Failures by week 2015 to 2016 ')
	axs[0,2].set_xticks(np.arange(0, 108, 8))
	axs[0,2].plot(range(len(event_week_soft_2015)+len(event_week_soft_2016)),list(event_week_soft_2015.values())+list(event_week_soft_2016.values()), 'b-', linewidth=1, label="2015-2016 Software failures")
	axs[0,2].plot(range(len(event_week_hard_2015)+len(event_week_hard_2016)),list(event_week_hard_2015.values())+list(event_week_hard_2016.values()), 'r:', linewidth=1, label="2015-2016 Hardware failures")
	axs[0,2].legend(framealpha=1,shadow=True, borderpad = 1, fancybox=True)
	
	################################################################################
	#PLOT CATEGORY AND REASON DATA
	################################################################################
	print("\nProcessing plots 4")
	

	barWidth = 0.15
	data_category_2015_2016_hardware = [event_category_2015['hardware'],event_category_2016['hardware']]
	data_category_2015_2016_software = [event_category_2016['software'],event_category_2016['software']]
	data_reason_2015_2016_user = [event_reason_2015['user'],event_reason_2016['user']]
	data_reason_2015_2016_system = [event_reason_2015['system'],event_reason_2016['system']]
	
	r1 = np.arange(1,3,1)
	r2 = [x + 0.012 + barWidth for x in r1]
	r3 = [x + 0.1 + barWidth for x in r2]
	r4 = [x + + 0.012 + barWidth for x in r3]

	ax = plt.gca()
	axs[1,0].bar(r1,data_category_2015_2016_hardware,width=barWidth, label="Hardware",color=['orangered'])
	axs[1,0].bar(r2,data_category_2015_2016_software,width=barWidth, label="Software",color=['red'])
	axs[1,0].bar(r3,data_reason_2015_2016_system,width=barWidth, label="System",color=['blue'])
	axs[1,0].bar(r4,data_reason_2015_2016_user,width=barWidth, label="User",color=['darkblue'])
	axs[1,0].set_xlabel('Years')
	axs[1,0].set_ylabel('Count of failures')
	axs[1,0].set_title('Failures by year')
	axs[1,0].set_xticklabels(['2015','2016'])
	axs[1,0].set_xticks([1.3,2.3])
	axs[1,0].legend(framealpha=1,shadow=True, borderpad = 1, fancybox=True, loc=1,prop={'size':7})

	plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
	plt.savefig("PLOT_category(week,year)_category-reason(year).pdf")
	print("\nPlot in file: <PLOT_category(week,year)_category-reason(year).pdf>")
	
	return 
	
	
if len(sys.argv) >= 2:
	dirName = sys.argv[1]
	#outputFileName = sys.argv[2]
	time_series(dirName, "outputFileName")
else:
	print ("ERROR, usage: %s <directory> <output file>" % sys.argv[0])
	print ("<directory>: Filter failure directory")
	#print ("<output file>: file name to output plot information")
	sys.exit(0)