
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
import seaborn as sns

	
def time_series(dir_name1, dir_name2, dir_name3, output_dir_name):
	
	file_jobs_hour_2014 = dir_name1 + "workload_hour_distribution_2014.txt"
	file_jobs_day_2014 = dir_name1 + "workload_day_distribution_2014.txt"
	file_jobs_month_2014 =  dir_name1 + "workload_month_distribution_2014.txt"	
	# file_jobs_day_month_2014 =  dir_name1 + "workload_day_month_distribution_2015.txt"	
	# file_jobs_day_year_2014 =  dir_name1 + "workload_day_year_distribution_2015.txt"	
	
	file_jobs_hour_2015 = dir_name2 + "workload_hour_distribution_2015.txt"
	file_jobs_day_2015 = dir_name2 + "workload_day_distribution_2015.txt"
	file_jobs_month_2015 =  dir_name2 + "workload_month_distribution_2015.txt"	
	file_jobs_day_month_2015 =  dir_name2 + "workload_day_month_distribution_2015.txt"	
	file_jobs_day_year_2015 =  dir_name2 + "workload_day_year_distribution_2015.txt"	
	
	file_jobs_hour_2016 = dir_name3 + "workload_hour_distribution_2016.txt"
	file_jobs_day_2016 = dir_name3 + "workload_day_distribution_2016.txt"
	file_jobs_month_2016 = dir_name3 + "workload_month_distribution_2016.txt"
	file_jobs_day_month_2016 = dir_name3 + "workload_day_month_distribution_2016.txt"
	file_jobs_day_year_2016 = dir_name3 + "workload_day_year_distribution_2016.txt"
	#############################
	
	file_fail_hour_2014 = dir_name1 + "failure_hour_distribution_2014.txt"
	file_fail_day_2014 = dir_name1 + "failure_day_distribution_2014.txt"
	file_fail_month_2014 = dir_name1 + "failure_month_distribution_2014.txt"
	# file_fail_day_month_2015 = dir_name1 + "failure_day_month_distribution_2015.txt"
	# file_fail_day_year_2015 = dir_name1 + "failure_day_year_distribution_2015.txt"
	
	file_fail_hour_2015 = dir_name2 + "failure_hour_distribution_2015.txt"
	file_fail_day_2015 = dir_name2 + "failure_day_distribution_2015.txt"
	file_fail_month_2015 = dir_name2 + "failure_month_distribution_2015.txt"
	file_fail_day_month_2015 = dir_name2 + "failure_day_month_distribution_2015.txt"
	file_fail_day_year_2015 = dir_name2 + "failure_day_year_distribution_2015.txt"
	
	
	file_fail_hour_2016 = dir_name3 + "failure_hour_distribution_2016.txt"
	file_fail_day_2016 = dir_name3 + "failure_day_distribution_2016.txt"
	file_fail_month_2016 = dir_name3 +  "failure_month_distribution_2016.txt"
	file_fail_day_month_2016 = dir_name3 +  "failure_day_month_distribution_2016.txt"
	file_fail_day_year_2016 = dir_name3 +  "failure_day_year_distribution_2016.txt"
	
	data_jobs_hour_2014 = []
	data_jobs_day_2014 = []
	data_jobs_month_2014 = []
	# data_jobs_day_month_2014 = []
	# data_jobs_day_year_2014 = []
	
	data_jobs_hour_2015 = []
	data_jobs_day_2015 = []
	data_jobs_month_2015 = []
	data_jobs_day_month_2015 = []
	data_jobs_day_year_2015 = []
	
	
	data_jobs_hour_2016 = []
	data_jobs_day_2016 = []
	data_jobs_month_2016 = []
	data_jobs_day_month_2016 = []
	data_jobs_day_year_2016 = []
	
	data_fail_hour_2014 = []
	data_fail_day_2014 = []
	data_fail_month_2014 = []
	# data_fail_day_month_2015 = []
	# data_fail_day_year_2015 = []
	
	data_fail_hour_2015 = []
	data_fail_day_2015 = []
	data_fail_month_2015 = []
	data_fail_day_month_2015 = []
	data_fail_day_year_2015 = []
	
	
	data_fail_hour_2016 = []
	data_fail_day_2016 = []
	data_fail_month_2016 = []
	data_fail_day_month_2016 = []
	data_fail_day_year_2016 = []
	
	#######################################################
	line_count = 0
	with open(file_jobs_hour_2014) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_jobs_hour_2014.append(int(item[1].strip()))
		#print(data_jobs_hour_2014)
	#######################################################
	#######################################################
	line_count = 0
	with open(file_jobs_hour_2015) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_jobs_hour_2015.append(int(item[1].strip()))
		#print(data_jobs_hour_2015)
	#######################################################
	line_count = 0
	with open(file_jobs_hour_2016) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_jobs_hour_2016.append(int(item[1].strip()))
		#print(data_jobs_hour_2016)
	
	######################################################
	line_count = 0
	with open(file_jobs_day_2014) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_jobs_day_2014.append(int(item[1].strip()))
		#print(data_jobs_day_2014)
	######################################################
	line_count = 0
	with open(file_jobs_day_2015) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_jobs_day_2015.append(int(item[1].strip()))
		#print(data_jobs_day_2015)
	
	#######################################################
	line_count = 0
	with open(file_jobs_day_2016) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_jobs_day_2016.append(int(item[1].strip()))
		#print(data_jobs_day_2016)
	#######################################################
	line_count = 0
	with open(file_jobs_day_month_2015) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_jobs_day_month_2015.append(int(item[1].strip()))
		#print(data_jobs_day_month_2015)
	
	#######################################################
	line_count = 0
	with open(file_jobs_day_month_2016) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_jobs_day_month_2016.append(int(item[1].strip()))
		#print(data_jobs_day_month_2016)
	# #######################################################
	line_count = 0
	with open(file_jobs_day_year_2015) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_jobs_day_year_2015.append(int(item[1].strip()))
		#print(data_jobs_day_year_2015)
	
	#######################################################
	line_count = 0
	with open(file_jobs_day_year_2016) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_jobs_day_year_2016.append(int(item[1].strip()))
		#print(data_jobs_day_year_2016)
		
	# #######################################################
	line_count = 0
	with open(file_jobs_month_2014) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_jobs_month_2014.append(int(item[1].strip()))
		#print(data_jobs_month_2014)	
	# #######################################################
	line_count = 0
	with open(file_jobs_month_2015) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_jobs_month_2015.append(int(item[1].strip()))
		#print(data_jobs_month_2015)
	#######################################################
	line_count = 0
	with open(file_jobs_month_2016) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_jobs_month_2016.append(int(item[1].strip()))
		#print(data_jobs_month_2016)
	#######################################################
	line_count = 0
	with open(file_fail_hour_2014) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_fail_hour_2014.append(int(item[1].strip()))
		#print(data_fail_hour_2014)
	#######################################################
	line_count = 0
	with open(file_fail_hour_2015) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_fail_hour_2015.append(int(item[1].strip()))
		#print(data_fail_hour_2015)
	#######################################################
	line_count = 0
	with open(file_fail_hour_2016) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_fail_hour_2016.append(int(item[1].strip()))
		#print(data_fail_hour_2016)
	######################################################
	line_count = 0
	with open(file_fail_day_2014) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_fail_day_2014.append(int(item[1].strip()))
		#print(data_fail_day_2014)
	######################################################
	line_count = 0
	with open(file_fail_day_2015) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_fail_day_2015.append(int(item[1].strip()))
		#print(data_fail_day_2015)
	#######################################################
	line_count = 0
	with open(file_fail_day_month_2015) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_fail_day_month_2015.append(int(item[1].strip()))
		#print(data_fail_day_month_2015)
	#######################################################
	line_count = 0
	with open(file_fail_day_month_2016) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_fail_day_month_2016.append(int(item[1].strip()))
		#print(data_fail_day_month_2016)
	# #######################################################
	line_count = 0
	with open(file_fail_day_year_2015) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_fail_day_year_2015.append(int(item[1].strip()))
		#print(data_fail_day_year_2015)
	#######################################################
	line_count = 0
	with open(file_fail_day_year_2016) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_fail_day_year_2016.append(int(item[1].strip()))
		#print(data_fail_day_year_2016)
	# #######################################################
	line_count = 0
	with open(file_fail_day_2016) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_fail_day_2016.append(int(item[1].strip()))
		#print(data_fail_day_2016)
	
	#######################################################
	line_count = 0
	with open(file_fail_month_2014) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_fail_month_2014.append(int(item[1].strip()))
		#print(data_fail_month_2014)
	#######################################################
	line_count = 0
	with open(file_fail_month_2015) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_fail_month_2015.append(int(item[1].strip()))
		#print(data_fail_month_2015)
	#######################################################
	line_count = 0
	with open(file_fail_month_2016) as log:
		line_count += 1
		if line_count == 1:
			next(log) 
		for line in log:		
			item = line.split()
			data_fail_month_2016.append(int(item[1].strip()))
		#print(data_fail_month_2016)
	
	# ############################################################
	# ############################################################
	# ############################################################
	print("Processing PLOT 1")
	
	fig = plt.figure()
	fig, axs = plt.subplots(3,3,figsize=(12, 8))
	
	ind = np.arange(1,13,1)                # the x locations for the groups
	w = 0.4
	axs[0,0].set_xticklabels( ['1','2','3','4','5','6','7','8','9','10','11','12'])
	axs[0,0].set_xticks([1.2,2.2,3.2,4.2,5.2,6.2,7.2,8.2,9.2,10.2,11.2,12.2])#, ['1','2','3','4','5','6','7','8','9','10','11','12'])#ind,('1','2','3','4','5','6','7','8','9','10','11','12'))
	axs[0,0].bar(ind,data_fail_month_2014,width=.4,color=['blue'])
	axs2 = axs[0,0].twinx()
	axs2.bar(ind+w,data_jobs_month_2014,width=.4,color=['red'])
	axs2.set_ylabel('Job Count',color='tab:red')
	axs2.tick_params(axis='y', labelcolor='tab:red',labelsize=8)
	axs[0,0].tick_params(axis = 'x',  labelsize = 8)
	axs[0,0].tick_params(axis='y', labelcolor='tab:blue',labelsize=8)
	axs[0,0].set_xlabel('Month')
	axs[0,0].set_ylabel('Failure Count',color='tab:blue')
	
	# ############################################################
	# ############################################################
	# ############################################################
	print("Processing PLOT 2")
	
	ind = np.arange(1,8,1)                # the x locations for the groups
	m = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
	list_names = []
	for i in m:
		list_names.append(i[:2])
	w = 0.4
	axs[0,1].set_xticklabels(list_names)
	axs[0,1].set_xticks([1.2,2.2,3.2,4.2,5.2,6.2,7.2])
	axs[0,1].bar(ind,data_fail_day_2014,width=.4,color=['blue'])
	
	axs2 = axs[0,1].twinx()
	axs2.bar(ind+w,data_jobs_day_2014,width=.4,color=['red'])
	axs2.set_ylabel('Job Count',color='tab:red' )
	axs2.tick_params(axis='y', labelcolor='tab:red',labelsize=8)
	axs[0,1].tick_params(axis = 'x',  labelsize = 8)
	
	#axs2.legend()
	axs[0,1].tick_params(axis='y', labelcolor='tab:blue',labelsize=8)
	#axs[0,0].legend()
	axs[0,1].set_xlabel('Day')
	axs[0,1].set_ylabel('Failure Count',color='tab:blue')
	
	# ############################################################
	# ############################################################
	# ############################################################
	print("Processing PLOT 3")
	
	ind = np.arange(1,25,1)                # the x locations for the groups
	w = 0.32
	axs[0,2].set_xticks([1.2,2.2,3.2,4.2,5.2,6.2,7.2])
	axs[0,2].bar(ind,data_fail_hour_2014,width=.3,color=['blue'])
	axs2 = axs[0,2].twinx()
	plt.xticks([1.2,2.2,3.2,4.2,5.2,6.2,7.2,8.2,9.2,10.2,11.2,12.2,13.2,14.2,15.2,16.2,17.2,18.2,19.2,20.2,21.2,22.2,23.2,24.2], ['1','','','4','','','7','','','10','','','13','','','16','','','19','','','22','',''])
	axs2.bar(ind+w,data_jobs_hour_2014,width=.3,color=['red'])
	axs2.set_ylabel('Job Count',color='tab:red')
	axs2.tick_params(axis='y', labelcolor='tab:red',labelsize=8)
	axs[0,2].tick_params(axis = 'x',  labelsize = 8)
	axs[0,2].tick_params(axis='y', labelcolor='tab:blue',labelsize=8)
	axs[0,2].set_xlabel('Hour')
	axs[0,2].set_ylabel('Failure Count',color='tab:blue')
	# ############################################################
	# ############################################################
	# ############################################################
	print("Processing PLOT 4")

	ind = np.arange(1,13,1)                # the x locations for the groups
	w = 0.4
	axs[1,0].set_xticklabels( ['1','2','3','4','5','6','7','8','9','10','11','12'])
	axs[1,0].set_xticks([1.2,2.2,3.2,4.2,5.2,6.2,7.2,8.2,9.2,10.2,11.2,12.2])#, ['1','2','3','4','5','6','7','8','9','10','11','12'])#ind,('1','2','3','4','5','6','7','8','9','10','11','12'))
	axs[1,0].bar(ind,data_fail_month_2015,width=.4,color=['blue'])
	axs2 = axs[1,0].twinx()
	axs2.bar(ind+w,data_jobs_month_2015,width=.4,color=['red'])
	axs2.set_ylabel('Job Count',color='tab:red')
	axs[1,0].tick_params(axis = 'x',  labelsize = 8)
	axs2.tick_params(axis='y', labelcolor='tab:red',labelsize=8)
	axs[1,0].tick_params(axis='y', labelcolor='tab:blue',labelsize=8)
	axs[1,0].set_xlabel('Month')
	axs[1,0].set_ylabel('Failure Count',color='tab:blue')
	
	
	# ############################################################
	# ############################################################
	# ############################################################
	print("Processing PLOT 5")
	
	ind = np.arange(1,8,1)                # the x locations for the groups
	m = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
	list_names = []
	for i in m:
		list_names.append(i[:2])
	w = 0.4
	axs[1,1].set_xticklabels( list_names)
	axs[1,1].set_xticks([1.2,2.2,3.2,4.2,5.2,6.2,7.2])
	axs[1,1].bar(ind,data_fail_day_2015,width=.4,color=['blue'])
	axs2 = axs[1,1].twinx()
	axs2.bar(ind+w,data_jobs_day_2015,width=.4,color=['red'])
	axs2.set_ylabel('Job Count',color='tab:red' )
	axs2.tick_params(axis='y', labelcolor='tab:red',labelsize=8)
	axs[1,1].tick_params(axis = 'x',  labelsize = 8)
	#axs2.legend()
	axs[1,1].tick_params(axis='y', labelcolor='tab:blue',labelsize=8)
	#axs[0,0].legend()
	axs[1,1].set_xlabel('Day')
	axs[1,1].set_ylabel('Failure Count',color='tab:blue')
	
	# ############################################################
	# ############################################################
	# ############################################################
	print("Processing PLOT 6")
	
	ind = np.arange(1,25,1)                # the x locations for the groups
	w = 0.32
	axs[1,2].set_xticks([1.2,2.2,3.2,4.2,5.2,6.2,7.2])
	axs[1,2].bar(ind,data_fail_hour_2015,width=.3,color=['blue'])
	axs2 = axs[1,2].twinx()
	plt.xticks([1.2,2.2,3.2,4.2,5.2,6.2,7.2,8.2,9.2,10.2,11.2,12.2,13.2,14.2,15.2,16.2,17.2,18.2,19.2,20.2,21.2,22.2,23.2,24.2], ['1','','','4','','','7','','','10','','','13','','','16','','','19','','','22','',''])
	axs2.bar(ind+w,data_jobs_hour_2015,width=.3,color=['red'])
	axs2.set_ylabel('Job Count',color='tab:red')
	axs[1,2].tick_params(axis = 'x',  labelsize = 8)	
	axs2.tick_params(axis='y', labelcolor='tab:red',labelsize=8)
	axs[1,2].tick_params(axis='y', labelcolor='tab:blue',labelsize=8)
	axs[1,2].set_xlabel('Hour')
	axs[1,2].set_ylabel('Failure Count',color='tab:blue')
	
	
	
	# ############################################################
	# ############################################################
	# ############################################################
	print("Processing PLOT 7")
	
	ind = np.arange(1,13,1)                # the x locations for the groups
	w = 0.4
	axs[2,0].set_xticklabels( ['1','2','3','4','5','6','7','8','9','10','11','12'])
	axs[2,0].set_xticks([1.2,2.2,3.2,4.2,5.2,6.2,7.2,8.2,9.2,10.2,11.2,12.2])#, ['1','2','3','4','5','6','7','8','9','10','11','12'])#ind,('1','2','3','4','5','6','7','8','9','10','11','12'))
	axs[2,0].bar(ind,data_fail_month_2016,width=.4,color=['blue'])
	axs2 = axs[2,0].twinx()
	axs2.bar(ind+w,data_jobs_month_2016,width=.4,color=['red'])
	axs2.set_ylabel('Job Count',color='tab:red')
	axs2.tick_params(axis='y', labelcolor='tab:red',labelsize=8)
	axs[2,0].tick_params(axis = 'x',  labelsize = 8)
	
	#axs2.legend()
	axs[2,0].tick_params(axis='y', labelcolor='tab:blue',labelsize=8)
	#axs[0,0].legend()
	axs[2,0].set_xlabel('Month')
	axs[2,0].set_ylabel('Failure Count',color='tab:blue')
	
	# ############################################################
	# ############################################################
	# ############################################################
	print("Processing PLOT 8")
	
	ind = np.arange(1,8,1)                # the x locations for the groups
	m = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
	list_names = []
	for i in m:
		list_names.append(i[:2])
	w = 0.4
	axs[2,1].set_xticklabels( list_names)
	axs[2,1].set_xticks([1.2,2.2,3.2,4.2,5.2,6.2,7.2])
	axs[2,1].bar(ind,data_fail_day_2016,width=.4,color=['blue'])
	axs2 = axs[2,1].twinx()
	axs2.bar(ind+w,data_jobs_day_2016,width=.4,color=['red'])
	axs2.set_ylabel('Job Count',color='tab:red')
	axs2.tick_params(axis='y', labelcolor='tab:red',labelsize=8)
	axs[2,1].tick_params(axis = 'x',  labelsize = 8)
	
	axs[2,1].tick_params(axis='y', labelcolor='tab:blue',labelsize=8)
	axs[2,1].set_xlabel('Day')
	axs[2,1].set_ylabel('Failure Count',color='tab:blue')
	
	# ############################################################
	# ############################################################
	# ############################################################
	print("Processing PLOT 9")
	
	ind = np.arange(1,25,1)                # the x locations for the groups
	w = 0.32
	axs[2,2].set_xticks([1.2,2.2,3.2,4.2,5.2,6.2,7.2])
	axs[2,2].bar(ind,data_fail_hour_2016,width=.3,color=['blue'])
	axs2 = axs[2,2].twinx()
	plt.xticks([1.2,2.2,3.2,4.2,5.2,6.2,7.2,8.2,9.2,10.2,11.2,12.2,13.2,14.2,15.2,16.2,17.2,18.2,19.2,20.2,21.2,22.2,23.2,24.2], ['1','','','4','','','7','','','10','','','13','','','16','','','19','','','22','',''])
	axs2.bar(ind+w,data_jobs_hour_2016,width=.3,color=['red'])
	axs2.set_ylabel('Job Count',color='tab:red')
	axs2.tick_params(axis='y', labelcolor='tab:red',labelsize=8)
	axs[2,2].tick_params(axis = 'x',  labelsize = 8)
	axs[2,2].tick_params(axis='y', labelcolor='tab:blue',labelsize=8)
	axs[2,2].set_xlabel('Hour')
	axs[2,2].set_ylabel('Failure Count',color='tab:blue')
	
	# #############################################################
	plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.85, hspace=0.35, wspace=0.60)
	plt.savefig(output_dir_name+"PLOT_correlation_jobs_failures.pdf")
	print("Plot in file: <"+output_dir_name+"PLOT_correlation_jobs_failures.pdf>")	
	
	
		
	########################################################################
	#correlations
	########################################################################
	
	print("Processing Pearson correlations plots")
	
	plt.clf()
	fig = plt.figure()
	fig, axs = plt.subplots(3,3,figsize=(14, 8))
	
	pearson_correlation_day_2014 = ss.stats.pearsonr(data_fail_day_2014, data_jobs_day_2014)
	pearson_correlation_hour_2014 = ss.stats.pearsonr(data_fail_hour_2014, data_jobs_hour_2014)
	pearson_correlation_month_2014 = ss.stats.pearsonr(data_fail_month_2014, data_jobs_month_2014)
	#pearson_correlation_day_month_2014 = ss.stats.pearsonr(data_fail_day_month_2014, data_jobs_day_month_2014)
	#pearson_correlation_day_year_2014 = ss.stats.pearsonr(data_fail_day_year_2014, data_jobs_day_year_2014)
	
	
	pearson_correlation_day_2015 = ss.stats.pearsonr(data_fail_day_2015, data_jobs_day_2015)
	pearson_correlation_hour_2015 = ss.stats.pearsonr(data_fail_hour_2015, data_jobs_hour_2015)
	pearson_correlation_month_2015 = ss.stats.pearsonr(data_fail_month_2015, data_jobs_month_2015)
	pearson_correlation_day_month_2015 = ss.stats.pearsonr(data_fail_day_month_2015, data_jobs_day_month_2015)
	pearson_correlation_day_year_2015 = ss.stats.pearsonr(data_fail_day_year_2015, data_jobs_day_year_2015)
	
	pearson_correlation_hour_2016 = ss.stats.pearsonr(data_fail_hour_2016, data_jobs_hour_2016)
	pearson_correlation_day_2016 = ss.stats.pearsonr(data_fail_day_2016, data_jobs_day_2016)
	pearson_correlation_month_2016 = ss.stats.pearsonr(data_fail_month_2016, data_jobs_month_2016)
	pearson_correlation_day_month_2016 = ss.stats.pearsonr(data_fail_day_month_2016, data_jobs_day_month_2016)
	pearson_correlation_day_year_2016 = ss.stats.pearsonr(data_fail_day_year_2016, data_jobs_day_year_2016)
	
	#correlate between years hour
	print("----------------------correlate between years hour-------------------------------")
	pearson_correlation_hour_jobs_2014_2015 = ss.stats.pearsonr(data_jobs_hour_2014, data_jobs_hour_2015)
	print("PC hour jobs 2014 - 2015: " + str(pearson_correlation_hour_jobs_2014_2015[0]))
	pearson_correlation_hour_fail_2014_2015 = ss.stats.pearsonr(data_fail_hour_2014, data_fail_hour_2015)
	print("PC hour fails 2014 - 2015: " + str(pearson_correlation_hour_fail_2014_2015[0]))
	print("----------------------")
	
	print("----------------------")
	pearson_correlation_hour_jobs_2015_2016 = ss.stats.pearsonr(data_jobs_hour_2015, data_jobs_hour_2016)
	print("PC hour jobs 2015 - 2016: " + str(pearson_correlation_hour_jobs_2015_2016[0]))
	pearson_correlation_hour_fail_2015_2016 = ss.stats.pearsonr(data_fail_hour_2015, data_fail_hour_2016)
	print("PC hour fails 2015 - 2016: " + str(pearson_correlation_hour_fail_2015_2016[0]))
	print("----------------------")
	
	print("----------------------")
	pearson_correlation_hour_jobs_2014_2016 = ss.stats.pearsonr(data_jobs_hour_2014, data_jobs_hour_2016)
	print("PC hour jobs 2014 - 2016: " + str(pearson_correlation_hour_jobs_2014_2016[0]))
	pearson_correlation_hour_fail_2014_2016 = ss.stats.pearsonr(data_fail_hour_2014, data_fail_hour_2016)
	print("PC hour fails 2014 - 2016: " + str(pearson_correlation_hour_fail_2014_2016[0]))
	print("----------------------")
	
	#correlate between years days
	print("----------------------correlate between years days------------------------------")
	pearson_correlation_day_jobs_2014_2015 = ss.stats.pearsonr(data_jobs_day_2014, data_jobs_day_2015)
	print("PC day jobs 2014 - 2015: " + str(pearson_correlation_day_jobs_2014_2015[0]))
	pearson_correlation_day_fail_2014_2015 = ss.stats.pearsonr(data_fail_day_2014, data_fail_day_2015)
	print("PC day fails 2014 - 2015: " + str(pearson_correlation_day_fail_2014_2015[0]))
	print("----------------------")
	
	print("----------------------")
	pearson_correlation_day_jobs_2015_2016 = ss.stats.pearsonr(data_jobs_day_2015, data_jobs_day_2016)
	print("PC day jobs 2015 - 2016: " + str(pearson_correlation_day_jobs_2015_2016[0]))
	pearson_correlation_day_fail_2015_2016 = ss.stats.pearsonr(data_fail_day_2015, data_fail_day_2016)
	print("PC day fails 2015 - 2016: " + str(pearson_correlation_day_fail_2015_2016[0]))
	print("----------------------")
	
	print("----------------------")
	pearson_correlation_day_jobs_2014_2016 = ss.stats.pearsonr(data_jobs_day_2014, data_jobs_day_2016)
	print("PC day jobs 2014 - 2016: " + str(pearson_correlation_hour_jobs_2014_2016[0]))
	pearson_correlation_day_fail_2014_2016 = ss.stats.pearsonr(data_fail_day_2014, data_fail_day_2016)
	print("PC day fails 2014 - 2016: " + str(pearson_correlation_day_fail_2014_2016[0]))
	print("----------------------")
	
	#correlate between years months
	print("----------------------correlate between years months------------------------------")
	pearson_correlation_month_jobs_2014_2015 = ss.stats.pearsonr(data_jobs_month_2014, data_jobs_month_2015)
	print("PC month jobs 2014 - 2015: " + str(pearson_correlation_month_jobs_2014_2015[0]))
	pearson_correlation_month_fail_2014_2015 = ss.stats.pearsonr(data_fail_month_2014, data_fail_month_2015)
	print("PC month fails 2014 - 2015: " + str(pearson_correlation_month_fail_2014_2015[0]))
	print("----------------------")
	
	print("----------------------")
	pearson_correlation_month_jobs_2015_2016 = ss.stats.pearsonr(data_jobs_month_2015, data_jobs_month_2016)
	print("PC month jobs 2015 - 2016: " + str(pearson_correlation_month_jobs_2015_2016[0]))
	pearson_correlation_month_fail_2015_2016 = ss.stats.pearsonr(data_fail_month_2015, data_fail_month_2016)
	print("PC month fails 2015 - 2016: " + str(pearson_correlation_month_fail_2015_2016[0]))
	print("----------------------")
	
	print("----------------------")
	pearson_correlation_month_jobs_2014_2016 = ss.stats.pearsonr(data_jobs_month_2014, data_jobs_month_2016)
	print("PC month jobs 2014 - 2016: " + str(pearson_correlation_hour_jobs_2014_2016[0]))
	pearson_correlation_month_fail_2014_2016 = ss.stats.pearsonr(data_fail_month_2014, data_fail_month_2016)
	print("PC month fails 2014 - 2016: " + str(pearson_correlation_month_fail_2014_2016[0]))
	print("----------------------")
	
	#linear regresion
	slope, intercept, r_value, p_value, std_err = ss.linregress(data_jobs_month_2014,data_fail_month_2014)
	
	#MONTH 2014
	ax = sns.regplot(x=data_jobs_month_2014, y=data_fail_month_2014, line_kws={"color":"r","alpha":0.7,"lw":2, "label":"$y=%3.7s*x+%3.7s$"%(slope, intercept)},ax = axs[0,2]);
	ax.legend()
	    
	axs[0,2].set_title('Correlation Workload-Failures by month 2014\n'+"PearC= "+str(round(pearson_correlation_month_2014[0],5))+" RS= "+str(round(r_value*2,5))+"-p="+str(round(p_value,5)),fontsize = 8)
	axs[0,2].set_ylabel('Failures')
	axs[0,2].set_xlabel('Workload')
	axs[0,2].tick_params(axis='y',labelsize=8)
	axs[0,2].tick_params(axis='x',labelsize=8)
	
	#linear regresion
	slope, intercept, r_value, p_value, std_err = ss.linregress(data_jobs_day_2014,data_fail_day_2014)
	
	#DAY 2014
	ax = sns.regplot(x=data_jobs_day_2014, y=data_fail_day_2014, line_kws={"color":"r","alpha":0.7,"lw":2, "label":"$y=%3.7s*x+%3.7s$"%(slope, intercept)},ax = axs[0,1])
	ax.legend()
	axs[0,1].set_title('Pearson correlation Workload-Failures by day 2014\n'+"PearC= "+str(round(pearson_correlation_day_2014[0],5))+" RS= "+str(round(r_value*2,5))+"-p="+str(round(p_value,5)),fontsize = 8)
	axs[0,1].set_ylabel('Failures')
	axs[0,1].set_xlabel('Workload')
	axs[0,1].tick_params(axis='y',labelsize=8)
	
	axs[0,1].tick_params(axis='x',labelsize=8)
	#linear regresion
	slope, intercept, r_value, p_value, std_err = ss.linregress(data_jobs_hour_2014,data_fail_hour_2014)
	
	#HOUR 2014
	ax = sns.regplot(x=data_jobs_hour_2014, y=data_fail_hour_2014, line_kws={"color":"r","alpha":0.7,"lw":2, "label":"$y=%3.7s*x+%3.7s$"%(slope, intercept)},ax = axs[0,0])
	ax.legend()
	axs[0,0].set_title('Pearson correlation Workload-Failures by hour 2014\n'+"PearC= "+str(round(pearson_correlation_hour_2014[0],5))+" RS= "+str(round(r_value*2,5))+"-p="+str(p_value),fontsize = 8)
	axs[0,0].set_ylabel('Failures')
	axs[0,0].set_xlabel('Workload')
	axs[0,0].tick_params(axis='y',labelsize=8)
	axs[0,0].tick_params(axis='x',labelsize=8)
	
	#######################################################################################
	#######################################################################################
	#2015
	
	#linear regresion
	slope, intercept, r_value, p_value, std_err = ss.linregress(data_jobs_month_2015,data_fail_month_2015)
	
	#MONTH 2015
	ax = sns.regplot(x=data_jobs_month_2015, y=data_fail_month_2015, line_kws={"color":"r","alpha":0.7,"lw":2, "label":"$y=%3.7s*x+%3.7s$"%(slope, intercept)},ax = axs[1,2]);
	ax.legend()
	axs[1,2].set_title('Correlation Workload-Failures by month 2015\n'+"PearC= "+str(round(pearson_correlation_month_2015[0],5))+" RS= "+str(round(r_value*2,5))+"-p="+str(round(p_value,5)),fontsize = 8)
	axs[1,2].set_ylabel('Failures')
	axs[1,2].set_xlabel('Workload')
	axs[1,2].tick_params(axis='y',labelsize=8)
	axs[1,2].tick_params(axis='x',labelsize=8)
	
	#linear regresion
	slope, intercept, r_value, p_value, std_err = ss.linregress(data_jobs_day_2015,data_fail_day_2015)
	
	#DAY 2015
	ax = sns.regplot(x=data_jobs_day_2015, y=data_fail_day_2015, line_kws={"color":"r","alpha":0.7,"lw":2, "label":"$y=%3.7s*x+%3.7s$"%(slope, intercept)},ax = axs[1,1])
	ax.legend()
	axs[1,1].set_title('Pearson correlation Workload-Failures by day 2015\n'+"PearC= "+str(round(pearson_correlation_day_2015[0],5))+" RS= "+str(round(r_value*2,5))+"-p="+str(round(p_value,5)),fontsize = 8)
	axs[1,1].set_ylabel('Failures')
	axs[1,1].set_xlabel('Workload')
	axs[1,1].tick_params(axis='y',labelsize=8)
	axs[1,1].tick_params(axis='x',labelsize=8)
	
	#linear regresion
	slope, intercept, r_value, p_value, std_err = ss.linregress(data_jobs_hour_2015,data_fail_hour_2015)
	
	#HOUR 2015
	ax = sns.regplot(x=data_jobs_hour_2015, y=data_fail_hour_2015, line_kws={"color":"r","alpha":0.7,"lw":2, "label":"$y=%3.7s*x+%3.7s$"%(slope, intercept)},ax = axs[1,0])
	ax.legend()
	axs[1,0].set_title('Pearson correlation Workload-Failures by hour 2015\n'+"PearC= "+str(round(pearson_correlation_hour_2015[0],5))+" RS= "+str(round(r_value*2,5))+"-p="+str(p_value),fontsize = 8)
	axs[1,0].set_ylabel('Failures')
	axs[1,0].set_xlabel('Workload')
	axs[1,0].tick_params(axis='y',labelsize=8)
	axs[1,0].tick_params(axis='x',labelsize=8)
	
	########################################################################################
	########################################################################################
	#2016
	
	#linear regresion
	slope, intercept, r_value, p_value, std_err = ss.linregress(data_jobs_month_2016,data_fail_month_2016)
	
	#MONTH 2016
	ax = sns.regplot(x=data_jobs_month_2016, y=data_fail_month_2016, line_kws={"color":"r","alpha":0.7,"lw":2, "label":"$y=%3.7s*x+%3.7s$"%(slope, intercept)},ax = axs[2,2])
	ax.legend()
	axs[2,2].set_title('Pearson correlation Workload-Failures by month 2016\n'+"PearC= "+str(round(pearson_correlation_month_2016[0],5))+" RS= "+str(round(r_value*2,5))+"-p="+str(p_value),fontsize = 8)
	axs[2,2].set_ylabel('Failures')
	axs[2,2].set_xlabel('Workload')
	axs[2,2].tick_params(axis='y',labelsize=8)
	axs[2,2].tick_params(axis='x',labelsize=8)
	
	#linear regresion
	slope, intercept, r_value, p_value, std_err = ss.linregress(data_jobs_day_2016,data_fail_day_2016)
	
	#DAY 2016
	ax = sns.regplot(x=data_jobs_day_2016, y=data_fail_day_2016, line_kws={"color":"r","alpha":0.7,"lw":2, "label":"$y=%3.7s*x+%3.7s$"%(slope, intercept)},ax = axs[2,1])
	ax.legend()
	axs[2,1].set_title('Pearson correlation Workload-Failures by day 2016\n'+"PearC= "+str(round(pearson_correlation_day_2016[0],5))+" RS= "+str(round(r_value*2,5))+"-p="+str(round(p_value,5)),fontsize = 8)
	axs[2,1].set_ylabel('Failures')
	axs[2,1].set_xlabel('Workload')
	axs[2,1].tick_params(axis='y',labelsize=8)
	axs[2,1].tick_params(axis='x',labelsize=8)
	
	#linear regresion
	slope, intercept, r_value, p_value, std_err = ss.linregress(data_jobs_hour_2016,data_fail_hour_2016)
	
	#HOUR 2016
	ax = sns.regplot(x=data_jobs_hour_2016, y=data_fail_hour_2016, line_kws={"color":"r","alpha":0.7,"lw":2, "label":"$y=%3.7s*x+%3.7s$"%(slope, intercept)},ax = axs[2,0])
	ax.legend()
	axs[2,0].set_title('Pearson correlation Workload-Failures by hour 2016\n'+"PearC= "+str(round(pearson_correlation_hour_2016[0],5))+" RS= "+str(round(r_value*2,5))+"-p="+str(round(p_value,5)),fontsize = 8)
	axs[2,0].set_ylabel('Failures')
	axs[2,0].set_xlabel('Workload')
	axs[2,0].tick_params(axis='y',labelsize=8)
	axs[2,0].tick_params(axis='x',labelsize=8)
	plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95, hspace=0.50, wspace=0.30)
	
	print("Plot in file: <"+output_dir_name+"Pearson_correlations.pdf>")	
	plt.savefig(output_dir_name+"Pearson_correlations.pdf")
	
	#################################################################
	plt.clf()
	fig = plt.figure()
	fig, axs = plt.subplots(2,3,figsize=(14, 8))
	
	#linear regresion
	slope, intercept, r_value, p_value, std_err = ss.linregress(data_jobs_day_month_2015,data_fail_day_month_2015)
	
	#DAY OF MONTH 2015
	ax = sns.regplot(x=data_jobs_day_month_2015, y=data_fail_day_month_2015, line_kws={"color":"r","alpha":0.7,"lw":2, "label":"$y=%3.7s*x+%3.7s$"%(slope, intercept)},ax = axs[0,0])
	ax.legend()
	axs[0,0].set_title('Pearson correlation Workload-Failures by day of month 2015\n'+"PearC= "+str(round(pearson_correlation_day_month_2015[0],5))+" RS= "+str(round(r_value*2,5))+"-p="+str(round(p_value,5)),fontsize = 8)
	axs[0,0].set_ylabel('Failures')
	axs[0,0].set_xlabel('Workload')
	axs[0,0].tick_params(axis='y',labelsize=8)
	axs[0,0].tick_params(axis='x',labelsize=8)
	
	
	#linear regresion
	slope, intercept, r_value, p_value, std_err = ss.linregress(data_jobs_day_month_2016,data_fail_day_month_2016)
	
	#DAY OF MONTH 2016
	ax = sns.regplot(x=data_jobs_day_month_2016, y=data_fail_day_month_2016, line_kws={"color":"r","alpha":0.7,"lw":2, "label":"$y=%3.7s*x+%3.7s$"%(slope, intercept)},ax = axs[0,1])
	ax.legend()
	axs[0,1].set_title('Pearson correlation Workload-Failures by day of month 2016\n'+"PearC= "+str(round(pearson_correlation_day_month_2016[0],5))+" RS= "+str(round(r_value*2,5))+"-p="+str(round(p_value,5)),fontsize = 8)
	axs[0,1].set_ylabel('Failures')
	axs[0,1].set_xlabel('Workload')
	axs[0,1].tick_params(axis='y',labelsize=8)
	axs[0,1].tick_params(axis='x',labelsize=8)
	
	#linear regresion
	slope, intercept, r_value, p_value, std_err = ss.linregress(data_jobs_day_year_2015,data_fail_day_year_2015)
	
	#DAY OF YEAR 2015
	ax = sns.regplot(x=data_jobs_day_year_2015, y=data_fail_day_year_2015, line_kws={"color":"r","alpha":0.7,"lw":2, "label":"$y=%3.7s*x+%3.7s$"%(slope, intercept)},ax = axs[1,0])
	ax.legend()
	axs[1,0].set_title('Pearson correlation Workload-Failures by day of year 2015\n'+"PearC= "+str(round(pearson_correlation_day_year_2015[0],5))+" RS= "+str(round(r_value*2,5))+"-p="+str(round(p_value,5)),fontsize = 8)
	axs[1,0].set_ylabel('Failures')
	axs[1,0].set_xlabel('Workload')
	axs[1,0].tick_params(axis='y',labelsize=8)
	axs[1,0].tick_params(axis='x',labelsize=8)
	
	
	#linear regresion
	slope, intercept, r_value, p_value, std_err = ss.linregress(data_jobs_day_year_2016,data_fail_day_year_2016)
	
	#DAY OF MONTH 2016
	ax = sns.regplot(x=data_jobs_day_year_2016, y=data_fail_day_year_2016, line_kws={"color":"r","alpha":0.7,"lw":2, "label":"$y=%3.7s*x+%3.7s$"%(slope, intercept)},ax = axs[1,1])
	ax.legend()
	axs[1,1].set_title('Pearson correlation Workload-Failures by day of year 2016\n'+"PearC= "+str(round(pearson_correlation_day_year_2016[0],5))+" RS= "+str(round(r_value*2,5))+"-p="+str(round(p_value,5)),fontsize = 8)
	axs[1,1].set_ylabel('Failures')
	axs[1,1].set_xlabel('Workload')
	axs[1,1].tick_params(axis='y',labelsize=8)
	axs[1,1].tick_params(axis='x',labelsize=8)
	
	
	plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95, hspace=0.30, wspace=0.20)
	print("Plot in file: <"+output_dir_name+"Pearson_correlations2.pdf>")	
	plt.savefig(output_dir_name+"Pearson_correlations2.pdf")
	
	plt.clf()
	ax = sns.jointplot(x=data_jobs_hour_2015, y=data_fail_hour_2015,marginal_kws=dict(bins=12), kind='reg', line_kws={"color":"r","alpha":0.7,"lw":2, "label":"$y=%3.7s*x+%3.7s$"%(slope, intercept)},space=0, height=6)

	
	print("Plot in file: <"+output_dir_name+"Pearson_correlations3.pdf>")	
	plt.savefig(output_dir_name+"Pearson_correlations3.pdf")
	
if len(sys.argv) >= 5:
	dir_year1 = sys.argv[1]
	dir_year2 = sys.argv[2]
	dir_year3 = sys.argv[3]
	output_dir_name = sys.argv[4]
	time_series(dir_year1, dir_year2,dir_year3, output_dir_name)
else:
	print ("ERROR, usage: %s <year 1 dir name><year 2 dir name><year 3 dir name><output dir name>" % sys.argv[0])
	print("Year 1 dir contains the result of: job_frequency_hourday_day_month.py and failures_frequency_hourday_day_month_By_one_Year.py for year 1")
	print("Year 2 dir contains the result of: job_frequency_hourday_day_month.py and failures_frequency_hourday_day_month_By_one_Year.py for year 2")
	print("Year 3 dir contains the result of: job_frequency_hourday_day_month.py and failures_frequency_hourday_day_month_By_one_Year.py for year 3")
	
	sys.exit(0)