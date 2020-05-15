
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
from datetime import date, timedelta, datetime as dt
import calendar as cl

def init_tables(event_day_year,event_week_daymonth, event_week_year,event_hour_day,event_hour_day_hard,event_hour_day_soft, event_week,event_week_hard,event_week_soft, event_month,event_month_hard,event_month_soft):
	""" Initializes tables """
	
	for i in range(365):
		if i not in event_day_year.keys():
			event_day_year[i] = 0
	
	event_week_daymonth[1] = 0
	event_week_daymonth[2] = 0
	event_week_daymonth[3] = 0
	event_week_daymonth[4] = 0
	event_week_daymonth[5] = 0
	event_week_daymonth[6] = 0
	event_week_daymonth[7] = 0
	event_week_daymonth[8] = 0
	event_week_daymonth[9] = 0
	event_week_daymonth[10] = 0
	event_week_daymonth[11] = 0
	event_week_daymonth[12] = 0
	event_week_daymonth[13] = 0
	event_week_daymonth[14] = 0
	event_week_daymonth[15] = 0
	event_week_daymonth[16] = 0
	event_week_daymonth[17] = 0
	event_week_daymonth[18] = 0
	event_week_daymonth[19] = 0
	event_week_daymonth[20] = 0
	event_week_daymonth[21] = 0
	event_week_daymonth[22] = 0
	event_week_daymonth[23] = 0
	event_week_daymonth[24] = 0
	event_week_daymonth[25] = 0
	event_week_daymonth[26] = 0
	event_week_daymonth[27] = 0
	event_week_daymonth[28] = 0
	event_week_daymonth[29] = 0
	event_week_daymonth[30] = 0
	event_week_daymonth[31] = 0

	event_hour_day_hard['00'] = 0
	event_hour_day_hard['01'] = 0
	event_hour_day_hard['02'] = 0
	event_hour_day_hard['03'] = 0
	event_hour_day_hard['04'] = 0
	event_hour_day_hard['05'] = 0
	event_hour_day_hard['06'] = 0
	event_hour_day_hard['07'] = 0
	event_hour_day_hard['08'] = 0
	event_hour_day_hard['09'] = 0
	event_hour_day_hard['10'] = 0
	event_hour_day_hard['11'] = 0
	event_hour_day_hard['12'] = 0
	event_hour_day_hard['13'] = 0
	event_hour_day_hard['14'] = 0
	event_hour_day_hard['15'] = 0
	event_hour_day_hard['16'] = 0
	event_hour_day_hard['17'] = 0
	event_hour_day_hard['18'] = 0
	event_hour_day_hard['19'] = 0
	event_hour_day_hard['20'] = 0
	event_hour_day_hard['21'] = 0
	event_hour_day_hard['22'] = 0
	event_hour_day_hard['23'] = 0
	event_hour_day_soft['00'] = 0
	event_hour_day_soft['01'] = 0
	event_hour_day_soft['02'] = 0
	event_hour_day_soft['03'] = 0
	event_hour_day_soft['04'] = 0
	event_hour_day_soft['05'] = 0
	event_hour_day_soft['06'] = 0
	event_hour_day_soft['07'] = 0
	event_hour_day_soft['08'] = 0
	event_hour_day_soft['09'] = 0
	event_hour_day_soft['10'] = 0
	event_hour_day_soft['11'] = 0
	event_hour_day_soft['12'] = 0
	event_hour_day_soft['13'] = 0
	event_hour_day_soft['14'] = 0
	event_hour_day_soft['15'] = 0
	event_hour_day_soft['16'] = 0
	event_hour_day_soft['17'] = 0
	event_hour_day_soft['18'] = 0
	event_hour_day_soft['19'] = 0
	event_hour_day_soft['20'] = 0
	event_hour_day_soft['21'] = 0
	event_hour_day_soft['22'] = 0
	event_hour_day_soft['23'] = 0

	event_hour_day['00'] = 0
	event_hour_day['01'] = 0
	event_hour_day['02'] = 0
	event_hour_day['03'] = 0
	event_hour_day['04'] = 0
	event_hour_day['05'] = 0
	event_hour_day['06'] = 0
	event_hour_day['07'] = 0
	event_hour_day['08'] = 0
	event_hour_day['09'] = 0
	event_hour_day['10'] = 0
	event_hour_day['11'] = 0
	event_hour_day['12'] = 0
	event_hour_day['13'] = 0
	event_hour_day['14'] = 0
	event_hour_day['15'] = 0
	event_hour_day['16'] = 0
	event_hour_day['17'] = 0
	event_hour_day['18'] = 0
	event_hour_day['19'] = 0
	event_hour_day['20'] = 0
	event_hour_day['21'] = 0
	event_hour_day['22'] = 0
	event_hour_day['23'] = 0
	
	event_week_soft['Monday'] = 0
	event_week_soft['Tuesday'] = 0
	event_week_soft['Wednesday'] = 0
	event_week_soft['Thursday'] = 0
	event_week_soft['Friday'] = 0
	event_week_soft['Saturday'] = 0
	event_week_soft['Sunday'] = 0
	
	event_week_hard['Monday'] = 0
	event_week_hard['Tuesday'] = 0
	event_week_hard['Wednesday'] = 0
	event_week_hard['Thursday'] = 0
	event_week_hard['Friday'] = 0
	event_week_hard['Saturday'] = 0
	event_week_hard['Sunday'] = 0
		
	event_week['Monday'] = 0
	event_week['Tuesday'] = 0
	event_week['Wednesday'] = 0
	event_week['Thursday'] = 0
	event_week['Friday'] = 0
	event_week['Saturday'] = 0
	event_week['Sunday'] = 0
	
	event_week_year[1] = 0
	event_week_year[2] = 0
	event_week_year[3] = 0
	event_week_year[4] = 0
	event_week_year[5] = 0
	event_week_year[6] = 0
	event_week_year[7] = 0
	event_week_year[8] = 0
	event_week_year[9] = 0
	event_week_year[10] = 0
	event_week_year[11] = 0
	event_week_year[12] = 0
	event_week_year[13] = 0
	event_week_year[14] = 0
	event_week_year[15] = 0
	event_week_year[16] = 0
	event_week_year[17] = 0
	event_week_year[18] = 0
	event_week_year[19] = 0
	event_week_year[20] = 0
	event_week_year[21] = 0
	event_week_year[22] = 0
	event_week_year[23] = 0
	event_week_year[24] = 0
	event_week_year[25] = 0
	event_week_year[26] = 0
	event_week_year[27] = 0
	event_week_year[28] = 0
	event_week_year[29] = 0
	event_week_year[30] = 0
	event_week_year[31] = 0
	event_week_year[32] = 0
	event_week_year[33] = 0
	event_week_year[34] = 0
	event_week_year[35] = 0
	event_week_year[36] = 0
	event_week_year[37] = 0
	event_week_year[38] = 0
	event_week_year[39] = 0
	event_week_year[40] = 0
	event_week_year[41] = 0
	event_week_year[42] = 0
	event_week_year[43] = 0
	event_week_year[44] = 0
	event_week_year[45] = 0
	event_week_year[46] = 0
	event_week_year[47] = 0
	event_week_year[48] = 0
	event_week_year[49] = 0
	event_week_year[50] = 0
	event_week_year[51] = 0
	event_week_year[52] = 0

	
	event_month_soft['01'] = 0
	event_month_soft['02'] = 0
	event_month_soft['03'] = 0
	event_month_soft['04'] = 0
	event_month_soft['05'] = 0
	event_month_soft['06'] = 0
	event_month_soft['07'] = 0
	event_month_soft['08'] = 0
	event_month_soft['09'] = 0
	event_month_soft['10'] = 0
	event_month_soft['11'] = 0
	event_month_soft['12'] = 0
	event_month_hard['01'] = 0
	event_month_hard['02'] = 0
	event_month_hard['03'] = 0
	event_month_hard['04'] = 0
	event_month_hard['05'] = 0
	event_month_hard['06'] = 0
	event_month_hard['07'] = 0
	event_month_hard['08'] = 0
	event_month_hard['09'] = 0
	event_month_hard['10'] = 0
	event_month_hard['11'] = 0
	event_month_hard['12'] = 0

	
	event_month['01'] = 0
	event_month['02'] = 0
	event_month['03'] = 0
	event_month['04'] = 0
	event_month['05'] = 0
	event_month['06'] = 0
	event_month['07'] = 0
	event_month['08'] = 0
	event_month['09'] = 0
	event_month['10'] = 0
	event_month['11'] = 0
	event_month['12'] = 0

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
	
def time_series(file_name, output_dir_name, year_text):
	#""" Reads a failure log file and correlates job IDs with MOAB log files in the directory """
	dayFormat = '%a_%b_%d_%Y'
	file_count = 0
	line_count = 0
	pathFileName = []
	event_hour_day = {}
	event_hour_day_hard = {}
	event_hour_day_soft = {}
	
	event_week = {}
	event_week_hard = {}
	event_week_soft = {}
	
	event_month = {}
	event_month_hard = {}
	event_month_soft = {}
	
	event_week_year = {}
	
	event_week_daymonth = {}
	event_day_year = {}
	
	format = '%Y-%m-%d %H:%M:%S'
	init = False
	startTime = time.clock()
    
	init_tables(event_day_year,event_week_daymonth,event_week_year,event_hour_day,event_hour_day_hard,event_hour_day_soft, event_week,event_week_hard,event_week_soft, event_month,event_month_hard,event_month_soft)
	
	line_count = 0
	print("\nPrcessing year: "+ year_text)

	with open(file_name) as log:
		line_count += 1
		if line_count == 1:
			next(log)
		for line in log:
			item = line.split("|")
			
			day_month = int(item[3].strip()[8:-9])
			
			
			week = int(datetime.date(int(item[3].strip()[:-15]), int(item[3].strip()[5:-12]), int(item[3].strip()[8:-9])).isocalendar()[1])
			#print(week)
			month = item[3].strip()[5:-12]
			year = item[3].strip()[:-15]
			#day of year
			day_year = int(datetime.datetime.strptime(year+"-"+month+"-"+str(day_month),"%Y-%m-%d").timetuple().tm_yday)
			 
			#Day of week
			f = dt.strptime(item[3].strip(),format)
			day_of_week = cl.day_name[f.weekday()] 
			#Hour of day
			hd = item[3].strip()
			hour_day = hd[11:-6]
			category = item[4].strip()
			reason = item[5].strip()
			
			if day_year in event_day_year.keys():
				event_day_year[day_year] += 1
						
			if day_month in event_week_daymonth.keys():
				event_week_daymonth[day_month] += 1
			
			if week in event_week_year.keys():
				event_week_year[week] += 1
			
			if hour_day in event_hour_day.keys():
				event_hour_day[hour_day] += 1
				if category == "hardware":
					event_hour_day_hard[hour_day] += 1
				if category == "software":
					event_hour_day_soft[hour_day] += 1
		
			if month in event_month.keys():
				event_month[month] += 1
				if category == "hardware":
					event_month_hard[month] += 1
				if category == "software":
					event_month_soft[month] += 1
		
			if day_of_week in event_week.keys():
				event_week[day_of_week] += 1
				if category == "hardware":
					event_week_hard[day_of_week] += 1
				if category == "software":
					event_week_soft[day_of_week] += 1
				continue
			
	# for save data of day year
	output_file_name = output_dir_name + '/' + 'failure_day_year_distribution_'+year_text+'.txt'
	output_file_txt = open(output_file_name, 'w')
	output_file_txt.write('DAY DAY_YEAR_FAILURES\n')
	print(event_week_daymonth)
	output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), range(365), event_day_year.values())))
	output_file_txt.close()
	
	# for save data of day month
	output_file_name = output_dir_name + '/' + 'failure_day_month_distribution_'+year_text+'.txt'
	output_file_txt = open(output_file_name, 'w')
	output_file_txt.write('DAY DAY_MONTH_FAILURES\n')
	output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), range(31), event_week_daymonth.values())))
	output_file_txt.close()
	
	# for save data of week
	output_file_name = output_dir_name + '/' + 'failure_week_year_distribution_'+year_text+'.txt'
	output_file_txt = open(output_file_name, 'w')
	output_file_txt.write('WEEK WEEK_YEAR_FAILURES\n')
	print(event_week_daymonth)
	output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), range(52), event_week_year.values())))
	output_file_txt.close()
	
	
	###########################################################################################3333
	#plt.style.use('seaborn-whitegrid')
	fig = plt.figure()
	fig, axs = plt.subplots(2,3,figsize=(13, 7))
		
	print("Processing failures graphic by month:  Graphic 1 of 6")	
	data = [event_month['01'],event_month['02'],event_month['03'],event_month['04'],event_month['05'],event_month['06'],event_month['07'],event_month['08'],event_month['09'],event_month['10'],event_month['11'],event_month['12']]
	axs[0,0].set_xticks(np.arange(1,13, 1))
	axs[0,0].bar(range(1,13,1),data,color=['blue'])
	plt.legend(framealpha=1,shadow=True, borderpad = 1, fancybox=True)
	axs[0,0].set_xlabel('Month')
	axs[0,0].set_ylabel('Failure Count')
	axs[0,0].set_title('Failures by Month' + year_text)
	# for save data
	output_file_name = output_dir_name + '/' + 'failure_month_distribution_'+year_text+'.txt'
	output_file_txt = open(output_file_name, 'w')
	output_file_txt.write('MONTH MONTH_FAILURES\n')
	output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), range(12), data)))
	output_file_txt.close()
	#############################################################
	print("Processing failures graphic by day:  Graphic 2 of 6")
	
	m = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
	list_names = []
	for i in m:
		list_names.append(i[:2])
	
	data = [event_week['Monday'],event_week['Tuesday'],event_week['Wednesday'],event_week['Thursday'],event_week['Friday'],event_week['Saturday'],event_week['Sunday']]
	axs[0,1].set_xticklabels(list_names)
	axs[0,1].set_xticks(np.arange(1,8, 1))
	axs[0,1].bar(range(1,8,1),data,color=['blue'])
	plt.legend(framealpha=1,shadow=True, borderpad = 1, fancybox=True)
	axs[0,1].set_xlabel('Day')
	axs[0,1].set_ylabel('Failure Count')
	axs[0,1].set_title('Failures by day of week ' + year_text)
	# for save data
	output_file_name = output_dir_name + '/' + 'failure_day_distribution_'+year_text+'.txt'
	output_file_txt = open(output_file_name, 'w')
	output_file_txt.write('DAY DAY_FAILURES\n')
	output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), range(7), data)))
	output_file_txt.close()
	
	############################################################
	print("Processing failures graphic by hour: Graphic 3 of 6")
	data = [event_hour_day['00'],event_hour_day['01'],event_hour_day['02'],event_hour_day['03'],event_hour_day['04'],event_hour_day['05'],event_hour_day['06'],event_hour_day['07'],event_hour_day['08'],event_hour_day['09'],event_hour_day['10'],event_hour_day['11'],event_hour_day['12'],event_hour_day['13'],event_hour_day['14'],event_hour_day['15'],event_hour_day['16'],event_hour_day['17'],event_hour_day['18'],event_hour_day['19'],event_hour_day['20'],event_hour_day['21'],event_hour_day['22'],event_hour_day['23']]	
	#axs[0,2].set_yticks(np.arange(0,max(event_hour_day.values())+5, 5))
	axs[0,2].bar(range(1,25,1),data,color=['blue'])
	plt.legend(framealpha=1,shadow=True, borderpad = 1, fancybox=True)
	axs[0,2].set_xlabel('Hour')
	axs[0,2].set_ylabel('Failure Count')
	axs[0,2].set_title('Failures by hour of day ' + year_text)
	plt.xticks([2.4,6.4,10.4,14.4,18.4,22.4], ['2','6','10','14','18','22'])
	# for save data
	output_file_name = output_dir_name + '/' + 'failure_hour_distribution_'+year_text+'.txt'
	output_file_txt = open(output_file_name, 'w')
	output_file_txt.write('HOUR HOUR_FAILURES\n')
	output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), range(24), data)))
	output_file_txt.close()
	
	############################################################
	############################################################
	############################################################
	#PLOT CATERGORY OF FAILURE(HARDWARE-SOFTWARE)
	print("Processing hardware-software failures graphic by month:  Graphic 4 of 6")
	n = 12
	ind = np.arange(1,13,1)                # the x locations for the groups
	w = 0.4
	data_hard = [event_month_hard['01'],event_month_hard['02'],event_month_hard['03'],event_month_hard['04'],event_month_hard['05'],event_month_hard['06'],event_month_hard['07'],event_month_hard['08'],event_month_hard['09'],event_month_hard['10'],event_month_hard['11'],event_month_hard['12']]
	data_soft = [event_month_soft['01'],event_month_soft['02'],event_month_soft['03'],event_month_soft['04'],event_month_soft['05'],event_month_soft['06'],event_month_soft['07'],event_month_soft['08'],event_month_soft['09'],event_month_soft['10'],event_month_soft['11'],event_month_soft['12']]

	axs[1,0].set_xticklabels( ['1','2','3','4','5','6','7','8','9','10','11','12'])
	axs[1,0].set_xticks([1.2,2.2,3.2,4.2,5.2,6.2,7.2,8.2,9.2,10.2,11.2,12.2])#, ['1','2','3','4','5','6','7','8','9','10','11','12'])#ind,('1','2','3','4','5','6','7','8','9','10','11','12'))
	axs[1,0].bar(ind,data_hard,width=.4, label="Hardware",color=['blue'])
	axs[1,0].bar(ind+w,data_soft,width=.4, label="Software",color=['red'])
	axs[1,0].legend(edgecolor="black")
	#plt.legend(framealpha=1,shadow=True, borderpad = 1, fancybox=True)
	axs[1,0].set_xlabel('Month')
	axs[1,0].set_ylabel('Failure Count')
	axs[1,0].set_title('Hardware and Software Failures by Month ' + year_text)
	fig.savefig('full_figure.png')
	# for save data
	output_file_name = output_dir_name + '/' + 'hardware_failure_month_distribution_'+year_text+'.txt'
	output_file_txt = open(output_file_name, 'w')
	output_file_txt.write('MONTH MONTH_HARD_FAILURES\n')
	output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), range(12), data_hard)))
	output_file_txt.close()
	
	output_file_name = output_dir_name + '/' + 'software_failure_month_distribution_'+year_text+'.txt'
	output_file_txt = open(output_file_name, 'w')
	output_file_txt.write('MONTH MONTH_SOFT_FAILURES\n')
	output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), range(12), data_soft)))
	output_file_txt.close()
	
	#############################################################
	print("Processing hardware-software failures graphic by day:  Graphic 5 of 6")
	n = 12
	ind = np.arange(1,8,1)                # the x locations for the groups
	w = 0.4
	
	m = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
	list_names = []
	for i in m:
		list_names.append(i[:2])
	
	data_soft = [event_week_soft['Monday'],event_week_soft['Tuesday'],event_week_soft['Wednesday'],event_week_soft['Thursday'],event_week_soft['Friday'],event_week_soft['Saturday'],event_week_soft['Sunday']]
	data_hard = [event_week_hard['Monday'],event_week_hard['Tuesday'],event_week_hard['Wednesday'],event_week_hard['Thursday'],event_week_hard['Friday'],event_week_hard['Saturday'],event_week_hard['Sunday']]
	
	axs[1,1].set_xticklabels(list_names)
	axs[1,1].set_xticks([1.2,2.2,3.2,4.2,5.2,6.2,7.2])#np.arange(1,8, 1))
	axs[1,1].bar(ind,data_hard,width=.4,label="Hardware",color=['blue'])
	axs[1,1].bar(ind+w,data_soft,width=.4,label="Software",color=['red'])
	
	axs[1,1].legend(edgecolor="black",prop={'size': 7})
	axs[1,1].set_xlabel('Day')
	axs[1,1].set_ylabel('Failure Count')
	axs[1,1].set_title('Hardware and Software Failures by day of week ' + year_text)
	# for save data
	output_file_name = output_dir_name + '/' + 'hardware_failure_day_distribution_'+year_text+'.txt'
	output_file_txt = open(output_file_name, 'w')
	output_file_txt.write('DAY DAY_HARD_FAILURES\n')
	output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), range(7), data_hard)))
	output_file_txt.close()
	
	output_file_name = output_dir_name + '/' + 'software_failure_day_distribution_'+year_text+'.txt'
	output_file_txt = open(output_file_name, 'w')
	output_file_txt.write('DAY DAY_SOFT_FAILURES\n')
	output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), range(7), data_soft)))
	output_file_txt.close()
	
	############################################################
	print("Processing hardware-software failures graphic by hour: Graphic 6 of 6")
	n = 12
	ind = np.arange(1,25,1)                # the x locations for the groups
	w = 0.4
	data_soft = [event_hour_day_soft['00'],event_hour_day_soft['01'],event_hour_day_soft['02'],event_hour_day_soft['03'],event_hour_day_soft['04'],event_hour_day_soft['05'],event_hour_day_soft['06'],event_hour_day_soft['07'],event_hour_day_soft['08'],event_hour_day_soft['09'],event_hour_day_soft['10'],event_hour_day_soft['11'],event_hour_day_soft['12'],event_hour_day_soft['13'],event_hour_day_soft['14'],event_hour_day_soft['15'],event_hour_day_soft['16'],event_hour_day_soft['17'],event_hour_day_soft['18'],event_hour_day_soft['19'],event_hour_day_soft['20'],event_hour_day_soft['21'],event_hour_day_soft['22'],event_hour_day_soft['23']]	
	data_hard = [event_hour_day_hard['00'],event_hour_day_hard['01'],event_hour_day_hard['02'],event_hour_day_hard['03'],event_hour_day_hard['04'],event_hour_day_hard['05'],event_hour_day_hard['06'],event_hour_day_hard['07'],event_hour_day_hard['08'],event_hour_day_hard['09'],event_hour_day_hard['10'],event_hour_day_hard['11'],event_hour_day_hard['12'],event_hour_day_hard['13'],event_hour_day_hard['14'],event_hour_day_hard['15'],event_hour_day_hard['16'],event_hour_day_hard['17'],event_hour_day_hard['18'],event_hour_day_hard['19'],event_hour_day_hard['20'],event_hour_day_hard['21'],event_hour_day_hard['22'],event_hour_day_hard['23']]

	axs[1,2].bar(ind,data_hard,width=.4,label="Hardware",color=['blue'])
	axs[1,2].bar(ind+w,data_soft,width=.4,label="Software",color=['red'])
	
	axs[1,2	].legend(edgecolor="black")
	axs[1,2].set_xlabel('Hour')
	axs[1,2].set_ylabel('Failure Count')
	axs[1,2].set_title('Hardware and Software Failures by hour of day ' + year_text)
	plt.xticks([2.3,6.3,10.3,14.3,18.3,22.3], ['2','6','10','14','18','22'])
	# for save data
	output_file_name = output_dir_name + '/' + 'hardare_failure_hour_distribution_'+year_text+'.txt'
	output_file_txt = open(output_file_name, 'w')
	output_file_txt.write('HOUR HOUR_HARD_FAILURES\n')
	output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), range(24), data_hard)))
	output_file_txt.close()
	
	output_file_name = output_dir_name + '/' + 'software_failure_hour_distribution_'+year_text+'.txt'
	output_file_txt = open(output_file_name, 'w')
	output_file_txt.write('HOUR HOUR_SOFT_FAILURES\n')
	output_file_txt.write('\n'.join(map(lambda x,y: str(x) + ' ' + str(y), range(24), data_soft)))
	output_file_txt.close()
	
	#############################################################
	plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
	plt.savefig("PLOT_failure_HourDay_week_month_"+year_text+".pdf")
	print("Plot in file: <PLOT_failure_HourDay_week_month_"+year_text+".pdf>")	
	
	
if len(sys.argv) >= 3:
	dir_name = sys.argv[1]
	output_dir_name = sys.argv[2]
	year = sys.argv[3]
	time_series(dir_name, output_dir_name, year)
else:
	print ("ERROR, usage: %s <directory> <output directory> <year>" % sys.argv[0])
	sys.exit(0)