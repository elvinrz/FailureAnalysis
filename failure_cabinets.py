
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
import seaborn as sns
from datetime import date, timedelta, datetime as dt
import calendar as cl

def init_tables(cabinet_2014,cabinet_2015,cabinet_2016,cabinet_2017,cabinet_2018, cabinet_week, cabinet_day,cabinet_total_count):
	""" Initializes tables """
	
	cabinet_week['Monday'] = {}
	cabinet_week['Tuesday'] = {}
	cabinet_week['Wednesday'] = {}
	cabinet_week['Thursday'] = {}
	cabinet_week['Friday'] = {}
	cabinet_week['Saturday'] = {}
	cabinet_week['Sunday'] = {}
	
	for i in range(1, 32):
		cabinet_day[i] = {}

	for i in range(25):
		cabinet_2014[i] = {}
		cabinet_2015[i] = {}
		cabinet_2016[i] = {}
		cabinet_2017[i] = {}
		cabinet_2018[i] = {}
		cabinet_total_count[i] = {}
	
def f_cabinets(dir_name):
	#""" Reads a failure log file and correlates job IDs with MOAB log files in the directory """
	file_count = 0
	line_count = 0
	pathFileName = []
	
	node_2015 = {}
	node_2015 = {}
	node_2016 = {}
	
	blade_2015 = {}
	blade_2016 = {}
	
	cabinet_day = {}
	cabinet_week = {}
	cabinet_2014 = {}
	cabinet_2015 = {}
	cabinet_2016 = {}
	cabinet_2017 = {}
	cabinet_2018 = {}
	
	cabinet_total_count = {}
	year_text = ""
	format = '%Y-%m-%d %H:%M:%S'
	
	
	init = False
	# start timer
	startTime = time.clock()
    #get all files of the year
	for path, dirs, files in os.walk(dir_name):
			for f in files:
				pathFileName.append(f)
	
	
	init_tables(cabinet_2014,cabinet_2015,cabinet_2016,cabinet_2017,cabinet_2018, cabinet_week,cabinet_day,cabinet_total_count)		
	
	# going through all files in directory
	for file_name in pathFileName:  
		file_count += 1
		line_count = 0
		print("\nPrcessing %s "% file_name)
		year_text = year_text +"_"+ str(2013 + file_count)
		count_event = 0
		file_name = dir_name + file_name
		
		
		with open(file_name) as log:
			line_count += 1
			if line_count == 1:
				next(log)
			for line in log:
				
				node_exist = True
			
				item = line.split("|")
				year = item[3].strip()[:-15]
				#day_name
				day = int(item[3].strip()[8:-9])
				#Day of week
				f = dt.strptime(item[3].strip(),format)
				day_of_week = cl.day_name[f.weekday()] 
				
				all_nodes = item[9].strip().split()
				#print("all nodes:" + str(all_nodes))
				for an in all_nodes: 
					cabinet = ""
					match = re.search(r"c(\d+)-(\d+)c(\d+)s(\d+)n(\d+)", an)
					if not match:
						match = re.search(r"c(\d+)-(\d+)c(\d+)s(\d+)", an)
						if match:
							cabinet = match.group(0)
					else:	
						cabinet =  match.group(0)
						
					#print(cabinet)
					if cabinet != "":
						cabinet = cabinet.split("-") #example c17-2xxxxxx
						cabinet_column = int(cabinet[0][1:].strip())  #extract c17
						cabinet_row = int(cabinet[1][:1]) #extract 2
						node = cabinet[1][-2:] #extract node number
						
						#count cabinets by day
						if cabinet_column not in cabinet_day[day]:
							cabinet_day[day][cabinet_column] = 1
						else:
							cabinet_day[day][cabinet_column] += 1
						
						#count cabinets by week
						if cabinet_column not in cabinet_week[day_of_week]:
							cabinet_week[day_of_week][cabinet_column] = 1
						else:
							cabinet_week[day_of_week][cabinet_column] += 1
						
						
						if cabinet_column not in cabinet_total_count:
							cabinet_total_count[cabinet_column] = {}	
						if cabinet_row not in cabinet_total_count[cabinet_column]:
							cabinet_total_count[cabinet_column][cabinet_row] = 1
						else:
							cabinet_total_count[cabinet_column][cabinet_row] += 1
						
						if year == "2014":
							if cabinet_column not in cabinet_2014:
								cabinet_2014[cabinet_column] = {}
							if cabinet_row not in cabinet_2014[cabinet_column]:
								cabinet_2014[cabinet_column][cabinet_row] = 1
							else:
								cabinet_2014[cabinet_column][cabinet_row] += 1
								continue
								
						if year == "2015":
							if cabinet_column not in cabinet_2015:
								cabinet_2015[cabinet_column] = {}
							if cabinet_row not in cabinet_2015[cabinet_column]:
								cabinet_2015[cabinet_column][cabinet_row] = 1
							else:
								cabinet_2015[cabinet_column][cabinet_row] += 1
								continue
					
						if year == "2016":
							if cabinet_column not in cabinet_2016:
								cabinet_2016[cabinet_column] = {}
							if cabinet_row not in cabinet_2016[cabinet_column]:
								cabinet_2016[cabinet_column][cabinet_row] = 1
							else:
								cabinet_2016[cabinet_column][cabinet_row] += 1
								continue
								
						if year == "2017":
							if cabinet_column not in cabinet_2017:
								cabinet_2017[cabinet_column] = {}
							if cabinet_row not in cabinet_2017[cabinet_column]:
								cabinet_2017[cabinet_column][cabinet_row] = 1
							else:
								cabinet_2017[cabinet_column][cabinet_row] += 1
								continue 
								
						if year == "2018":
							if cabinet_column not in cabinet_2018:
								cabinet_2018[cabinet_column] = {}
							if cabinet_row not in cabinet_2018[cabinet_column]:
								cabinet_2018[cabinet_column][cabinet_row] = 1
							else:
								cabinet_2018[cabinet_column][cabinet_row] += 1	
	#control data
	print("2014")
	print(cabinet_2014)	
	print("////////////////////////////////////////////////////////////")
	print("2015")
	print(cabinet_2015)
	print("////////////////////////////////////////////////////////////")
	print("2016")
	print(cabinet_2016)
	print("////////////////////////////////////////////////////////////")
	print("2017")
	print(cabinet_2017)
	print("////////////////////////////////////////////////////////////")
	print("2018")
	print(cabinet_2018)
	print("////////////////////////////////////////////////////////////")
	print("Total")
	print(cabinet_total_count)
	print("////////////////////////////////////////////////////////////")
	
	
	#Matrix to contanis data 
	cabinets_matrix_all_years = np.zeros((8, 25))
	cabinets_matrix_2014 = np.zeros((8, 25))
	cabinets_matrix_2015 = np.zeros((8, 25))
	cabinets_matrix_2016 = np.zeros((8, 25))
	cabinets_matrix_2017 = np.zeros((8, 25))
	cabinets_matrix_2018 = np.zeros((8, 25))
	
	cabinets_matrix_week = np.zeros((7, 25))
	cabinets_matrix_day = np.zeros((31, 25))
	
	rtotal = r2014 = r2015 = r2016 = r2017 = r2018 = 0
	#Total count
	for i,c1 in zip(range(25), cabinet_total_count.keys()):
		for j,c2 in zip(range(8),cabinet_total_count[c1].keys()):
			cabinets_matrix_all_years[j][i]  = cabinet_total_count[c1][c2]
			rtotal += cabinet_total_count[c1][c2]	
			#print("["+str(c1)+"-"+str(c2)+"]" + str(cabinet_total_count[c1][c2]))
	
	print("Total: "+str(rtotal))
	print("________________________________________________")
	#2014 count
	for i,c1 in zip(range(25), cabinet_2014.keys()):
		for j,c2 in zip(range(8),cabinet_2014[c1].keys()):
			cabinets_matrix_2014[j][i]  = cabinet_2014[c1][c2]
			r2014 += cabinet_2014[c1][c2]
			#print("["+str(c1)+"-"+str(c2)+"]" + str(cabinet_2014[c1][c2]))
	print("Total 2014: "+ str(r2014))
	print("________________________________________________")
	
	#2015 count
	for i,c1 in zip(range(25), cabinet_2015.keys()):
		for j,c2 in zip(range(8),cabinet_2015[c1].keys()):
			cabinets_matrix_2015[j][i]  = cabinet_2015[c1][c2]
			r2015 += cabinet_2015[c1][c2]	
			#print("["+str(c1)+"-"+str(c2)+"]" + str(cabinet_2015[c1][c2]))
			
	print("Total 2015: "+str(r2015))
	print("________________________________________________")
	
	#2016 count
	for i,c1 in zip(range(25), cabinet_2016.keys()):
		for j,c2 in zip(range(8),cabinet_2016[c1].keys()):
			cabinets_matrix_2016[j][i]  = cabinet_2016[c1][c2]
			r2016 += cabinet_2016[c1][c2]	
			#print("["+str(c1)+"-"+str(c2)+"]" + str(cabinet_2016[c1][c2]))
	print("Total 2016: "+str(r2016))
	print("________________________________________________")
	#2017 count
	for i,c1 in zip(range(25), cabinet_2017.keys()):
		for j,c2 in zip(range(8),cabinet_2017[c1].keys()):
			cabinets_matrix_2017[j][i]  = cabinet_2017[c1][c2]
			r2017 += cabinet_2017[c1][c2]	
			#print("["+str(c1)+"-"+str(c2)+"]" + str(cabinet_2017[c1][c2]))
	print("Total 2017: "+str(r2017))
	print("________________________________________________")
	#2018 count
	for i,c1 in zip(range(25), cabinet_2018.keys()):
		for j,c2 in zip(range(8),cabinet_2018[c1].keys()):
			cabinets_matrix_2018[j][i]  = cabinet_2018[c1][c2]
			r2018 += cabinet_2018[c1][c2]	
			#print("["+str(c1)+"-"+str(c2)+"]" + str(cabinet_2018[c1][c2]))
	print("Total 2018: "+str(r2018))
	print("________________________________________________")
	# print("WEEK////////////////////////////////////////////////////////////")		
	# print(cabinet_week)
	# print("////////////////////////////////////////////////////////////")
	
	#total count cabinet failures by week
	for c1 in cabinet_week.keys():
		if c1 == "Monday":
			for j,c2 in zip(range(25),cabinet_week[c1].keys()):
				cabinets_matrix_week[0][j]  = cabinet_week[c1][c2]
		if c1 == "Tuesday":
			for j,c2 in zip(range(25),cabinet_week[c1].keys()):
				cabinets_matrix_week[1][j]  = cabinet_week[c1][c2]
		if c1 == "Wednesday":
			for j,c2 in zip(range(25),cabinet_week[c1].keys()):
				cabinets_matrix_week[2][j]  = cabinet_week[c1][c2]
		if c1 == "Thursday":
			for j,c2 in zip(range(25),cabinet_week[c1].keys()):
				cabinets_matrix_week[3][j]  = cabinet_week[c1][c2]
		if c1 == "Friday":
			for j,c2 in zip(range(25),cabinet_week[c1].keys()):
				cabinets_matrix_week[4][j]  = cabinet_week[c1][c2]
		if c1 == "Saturday":
			for j,c2 in zip(range(25),cabinet_week[c1].keys()):
				cabinets_matrix_week[5][j]  = cabinet_week[c1][c2]
		if c1 == "Sunday":
			for j,c2 in zip(range(25),cabinet_week[c1].keys()):
				cabinets_matrix_week[6][j]  = cabinet_week[c1][c2]
	
	#print("MATRIX WEEK////////////////////////////////////////////////////////////")
	#print(cabinets_matrix_week)
	# for i,c1 in zip(range(7), cabinet_week.keys()):
		# for j,c2 in zip(range(25),cabinet_week[c1].keys()):
			# cabinets_matrix_week[i][j]  = cabinet_week[c1][c2]
			
	#print("DAY////////////////////////////////////////////////////////////")
	#print(cabinet_day)
	#print("////////////////////////////////////////////////////////////")
	
	#total count cabinet failures by day
	for i,c1 in zip(range(31), cabinet_day.keys()):
		for j,c2 in zip(range(25),cabinet_day[c1].keys()):
			cabinets_matrix_day[i][j]  = cabinet_day[c1][c2]
		
	#################################################################
	#PLOT count of cabinets
	#plt.rc('font',family='Times New Roman')
	
	fig = plt.figure()
	fig, axs = plt.subplots(2,3,figsize=(15, 8))
	
	sns.set()
	rows = ["0","1","2","3","4","5","6","7"]
	cols = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24"]
	
	#plot heatmap count 2014
	print("\nPrcessing PLOT 0")
	ax = sns.heatmap(cabinets_matrix_2014,cmap="coolwarm", linewidths=0.1, ax = axs[0,0])
	ax.invert_yaxis()
	axs[0,0].axis("tight")
	axs[0,0].set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,24.5])
	axs[0,0].set_yticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
	axs[0,0].set_xticklabels(cols,fontsize=7)
	axs[0,0].set_yticklabels(rows)
	axs[0,0].set_title('2014')
	axs[0,0].set_ylabel('Cabinet Row')
	axs[0,0].set_xlabel('Cabinet Column')
	
	
	#plot heatmap count 2015
	print("\nPrcessing PLOT 1")
	ax = sns.heatmap(cabinets_matrix_2015,cmap="coolwarm", linewidths=0.1, ax = axs[0,1])
	ax.invert_yaxis()
	axs[0,1].axis("tight")
	axs[0,1].set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,24.5])
	axs[0,1].set_yticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
	axs[0,1].set_xticklabels(cols,fontsize=7)
	axs[0,1].set_yticklabels(rows)
	axs[0,1].set_title('2015')
	axs[0,1].set_ylabel('Cabinet Row')
	axs[0,1].set_xlabel('Cabinet Column')
	
	#plot heatmap count 2016
	print("\nPrcessing PLOT 2")
	ax = sns.heatmap(cabinets_matrix_2016,cmap="coolwarm", linewidths=0.1, ax = axs[0,2])
	ax.invert_yaxis()
	axs[0,2].axis("tight")
	axs[0,2].set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,24.5])
	axs[0,2].set_yticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
	axs[0,2].set_xticklabels(cols,fontsize=7)
	axs[0,2].set_yticklabels(rows)
	axs[0,2].set_title('2016')
	axs[0,2].set_ylabel('Cabinet Row')
	axs[0,2].set_xlabel('Cabinet Column')
	
	#plot heatmap count 2017
	print("\nPrcessing PLOT 3")
	ax = sns.heatmap(cabinets_matrix_2017,cmap="coolwarm", linewidths=0.1, ax = axs[1,0])
	ax.invert_yaxis()
	axs[1,0].axis("tight")
	axs[1,0].set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,24.5])
	axs[1,0].set_yticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
	axs[1,0].set_xticklabels(cols,fontsize=7)
	axs[1,0].set_yticklabels(rows)
	axs[1,0].set_title('2017')
	axs[1,0].set_ylabel('Cabinet Row')
	axs[1,0].set_xlabel('Cabinet Column')
	
	#plot heatmap count 2018
	print("\nPrcessing PLOT 4")
	ax = sns.heatmap(cabinets_matrix_2018,cmap="coolwarm", linewidths=0.1, ax = axs[1,1])
	ax.invert_yaxis()
	axs[1,1].axis("tight")
	axs[1,1].set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,24.5])
	axs[1,1].set_yticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
	axs[1,1].set_xticklabels(cols,fontsize=7)
	axs[1,1].set_yticklabels(rows)
	axs[1,1].set_title('2018')
	axs[1,1].set_ylabel('Cabinet Row')
	axs[1,1].set_xlabel('Cabinet Column')
	
	#plot heatmap count 2014-2018
	print("\nPrcessing PLOT 5")		
	ax = sns.heatmap(cabinets_matrix_all_years,cmap="coolwarm", linewidths=0.1, ax = axs[1,2])
	ax.invert_yaxis()
	axs[1,2].axis("tight")
	axs[1,2].set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,24.5])
	axs[1,2].set_yticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
	axs[1,2].set_xticklabels(cols,fontsize=7)
	axs[1,2].set_yticklabels(rows)
	axs[1,2].set_title('2014-2015-2016-2017-2018')
	axs[1,2].set_ylabel('Cabinet Row')
	axs[1,2].set_xlabel('Cabinet Column')
	
	
	# #plot heatmap count 2015-2016 cabinet failures by week
	# print("\nPrcessing PLOT 4")
	# rows = ["Mon","Tue","wed","Thu","Fri","Sat","Sun"]
	# ax = sns.heatmap(cabinets_matrix_week,cmap="coolwarm", linewidths=0.1, ax = axs[1,1])
	# ax.invert_yaxis()
	# axs[1,1].axis("tight")
	# axs[1,1].set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,24.5])
	# axs[1,1].set_yticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5])
	# axs[1,1].set_xticklabels(cols,fontsize=7)
	# axs[1,1].set_yticklabels(rows)
	# axs[1,1].set_title('2015-2016 by week')
	# axs[1,1].set_ylabel('Day of week')
	# axs[1,1].set_xlabel('Cabinet Column')
	
	# #plot heatmap count 2015-2016 cabinet failures by day
	# print("\nPrcessing PLOT 5")
	# rows = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"]
	# ax = sns.heatmap(cabinets_matrix_day,cmap="coolwarm", linewidths=0.1, ax = axs[1,2])
	# ax.invert_yaxis()
	# axs[1,2].axis("tight")
	# axs[1,2].set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,24.5])
	# axs[1,2].set_yticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,24.5,25.5,26.5,27.5,28.5,29.5,30.5,31.5])
	# axs[1,2].set_xticklabels(cols,fontsize=7)
	# axs[1,2].set_yticklabels(rows,fontsize=7)
	# axs[1,2].set_title('2015-2016 by day')
	# axs[1,2].set_ylabel('Day')
	# axs[1,2].set_xlabel('Cabinet Column')
	
	fig.tight_layout()
	plt.savefig("PLOT_failure_cabinets_" + year_text +".pdf")
	print("\nPlot in file: <PLOT_failure_cabinets"+ year_text +".pdf>")
	
	########################################################################################
	#plot CLUSTERMAP count 2015-2016 cabinet failures by day
	
	# print("\nPrcessing CLUSTERMAP")
	# plt.clf()
	# fig = plt.figure()
	# fig, axs = plt.subplots(2,3,figsize=(7, 4))
	# ax = sns.clustermap(cabinets_matrix_two_years,cmap="coolwarm", linewidths=0.1)
	# axs[0,0].axis("tight")
	# axs[0,0].set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5,22.5,23.5,24.5])
	# axs[0,0].set_yticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
	# axs[0,0].set_xticklabels(cols,fontsize=7)
	# axs[0,0].set_yticklabels(rows)
	# axs[0,0].set_title('2015-2016')
	# axs[0,0].set_ylabel('Cabinet row')
	# axs[0,0].set_xlabel('Cabinet Column')
	
	# fig.tight_layout()
	# plt.savefig("PLOT_CLUSTERMAP_failure_cabinets_" + year_text +".pdf")
	# print("\nPlot in file: <PLOT_CLUSTERMAP_failure_cabinets"+ year_text +".pdf>")
	
	# ##############################################################################
	return 
	
	
if len(sys.argv) >= 2:
	dirName = sys.argv[1]
	outputFileName = "" #sys.argv[2]
	f_cabinets(dirName)
else:
	print ("ERROR, usage: %s <directory> <output file>" % sys.argv[0])
	sys.exit(0)