
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
from matplotlib import pylab
import numpy as np
import scipy.stats as ss
from collections import defaultdict
import collections
from datetime import date, timedelta
import seaborn as sns
from datetime import date, timedelta, datetime as dt
import calendar as cl
#import squarify
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

def init_tables(hardware_failures_clasification,software_failures_clasification):
	# """ Initializes tables """

	hardware_failures_clasification["GPU"] = {}
	hardware_failures_clasification["GPU"]["GPU DBE"] = 0	
	hardware_failures_clasification["GPU"]["GPU DPR"] = 0
	hardware_failures_clasification["Processor"] = {}
	hardware_failures_clasification["Processor"]["Machine Check Exception"] = 0
	hardware_failures_clasification["Blade"] = {}
	hardware_failures_clasification["Blade"]["Voltage Fault"] = 0
	hardware_failures_clasification["Blade"]["Module Failed"] = 0
	
	software_failures_clasification["GPU"] = {}
	software_failures_clasification["GPU"]["GPU BUS"] = 0
	software_failures_clasification["GPU"]["GPU XID"] = 0
	software_failures_clasification["Memory"] = {}
	software_failures_clasification["Memory"]["Out of Memory"] = 0
	software_failures_clasification["Memory"]["Kernel Panic"] = 0
	software_failures_clasification["Memory"]["RDMA Failure"] = 0
	software_failures_clasification["Memory"]["LBUG"] = 0
	
def f_cabinets(dir_name):
	#""" Reads a failure log file and correlates job IDs with MOAB log files in the directory """
	file_count = 0
	line_count = 0
	pathFileName = []
	count_nodes = 0 
	failures_node_affected_soft = []
	failures_node_affected_hard = []
	failures_node_affected = []
	ftotal = []
	fsoft = []
	fhard = []
	hardware_failures_clasification = {}
	software_failures_clasification = {}
	
	for i in range(5):
		failures_node_affected.append(0)
		failures_node_affected_hard.append(0)
		failures_node_affected_soft.append(0)
		ftotal.append(0)
		fhard.append(0)
		fsoft.append(0)
		
	year_text = ""
	format = '%Y-%m-%d %H:%M:%S'
	
	
	init = False
	# start timer
	startTime = time.clock()
    #get all files of the year
	for path, dirs, files in os.walk(dir_name):
			for f in files:
				pathFileName.append(f)
	
	
	init_tables(hardware_failures_clasification,software_failures_clasification)		
	
	# going through all files in directory
	for file_name in pathFileName:  
		file_count += 1
		line_count = 0
		print("\nPrcessing %d year, File name: %s "% (file_count, file_name))
		year_text = year_text +"_"+ str(2013 + file_count)
		count_event = 0
		file_name = dir_name + file_name
		
		
		with open(file_name) as log:
			line_count += 1
			if line_count == 1:
				next(log)
			for line in log:
				cabinet = ""
				node_exist = True
			
				item = line.split("|")
				year = item[3].strip()[:-15]
				#day_name
				day = int(item[3].strip()[8:-9])
				#Day of week
				f = dt.strptime(item[3].strip(),format)
				day_of_week = cl.day_name[f.weekday()] 
				category = item[4].strip() 
				description = item[6].strip()
				all_nodes = item[9].strip().split()
				
				if category == "hardware":
					if description == "GPU DBE":
						hardware_failures_clasification["GPU"]["GPU DBE"] += 1		
					if description == "GPU DPR":
						hardware_failures_clasification["GPU"]["GPU DPR"] += 1		
					if description == "Machine Check Exception":
						hardware_failures_clasification["Processor"]["Machine Check Exception"] += 1		
					if description == "Voltage Fault":
						hardware_failures_clasification["Blade"]["Voltage Fault"] += 1		
					if description == "Module Failed":
						hardware_failures_clasification["Blade"]["Module Failed"] += 1		
				
				if category == "software":
					if description == "GPU BUS":
						software_failures_clasification["GPU"]["GPU BUS"] += 1		
					if description == "GPU XID":
						software_failures_clasification["GPU"]["GPU XID"] += 1	
					if description == "Out of Memory":
						software_failures_clasification["Memory"]["Out of Memory"] += 1		
					if description == "Kernel Panic":
						software_failures_clasification["Memory"]["Kernel Panic"] += 1		
					if description == "RDMA Failure":
						software_failures_clasification["Memory"]["RDMA Failure"] += 1		
					if description == "LBUG":
						software_failures_clasification["Memory"]["LBUG"] += 1	
				
				#################################################################################
				#count all nodes
				count_nodes += len(all_nodes)
				t = len(all_nodes)
				#count nodes by category				
				if (t == 1):
					failures_node_affected[0] += 1
					ftotal[0] += 1
					if category == "hardware":
						failures_node_affected_hard[0] += 1
						fhard[0] += 1
					if category == "software":
						failures_node_affected_soft[0] += 1
						fsoft[0] += 1
						
				if (t == 2):
					failures_node_affected[1] += 1
					ftotal[1] += 1
					if category == "hardware":
						failures_node_affected_hard[1] += 1
						fhard[1] += 1
					if category == "software":
						failures_node_affected_soft[1] += 1
						fsoft[1] += 1
						
				if (t == 3):
					failures_node_affected[2] += 1
					ftotal[2] += 1
					if category == "hardware":
						failures_node_affected_hard[2] += 1
						fhard[2] += 1
					if category == "software":
						failures_node_affected_soft[2] += 1
						fsoft[2] += 1
				if (t == 4):
					failures_node_affected[3] += 1
					ftotal[3] += 1
					if category == "hardware":
						failures_node_affected_hard[3] += 1
						fhard[3] += 1
					if category == "software":
						failures_node_affected_soft[3] += 1
						fsoft[3] += 1
						
				if (t > 4):
					failures_node_affected[4] += 1
					ftotal[4] += 1
					if category == "hardware":
						failures_node_affected_hard[4] += 1
						fhard[4] += 1
					if category == "software":
						failures_node_affected_soft[4] += 1
						fsoft[4] += 1
			#print(hardware_failures_clasification)
			
			#print(software_failures_clasification)
			
			print("Year: "+ year)	
			
			print("Total: "+str(failures_node_affected))
			print("Hard: "+str(failures_node_affected_hard))
			print("Soft: "+str(failures_node_affected_soft))
			print("............")
			print("Total year: "+str(ftotal))
			print("Hard year: "+str(fhard))
			print("Soft year: "+str(fsoft))
			
			ftotal.clear()
			fhard.clear()
			fsoft.clear()
			
			for i in range(5):
				ftotal.append(0)
				fhard.append(0)
				fsoft.append(0)
			
			
			print(str(count_nodes))
			print("----------------------------------------------")
			count_nodes = 0
		
	#################################################################
	#PLOT count of cabinets
	
	fig = plt.figure()
	fig, axs = plt.subplots(2,3,figsize=(15, 8))
	
	plt.rc('font', family='DejaVu Sans')
	plt.rc('font', serif='Times New Roman')
	
	#sns.set()
	barWidth = 0.3
	r1 = np.arange(1,6,1)
	r2 = [x + 0.02 + barWidth for x in r1]
	
	ax = plt.gca()
	axs[0,0].bar(r1,failures_node_affected_hard,width=barWidth, label="Hardware",color=['blue'],log=True)
	axs[0,0].bar(r2,failures_node_affected_soft,width=barWidth, label="Software",color=['red'],log=True)
	axs[0,0].set_xlabel('Number of Nodes Simultaneously Affected')
	axs[0,0].set_ylabel('Failure Count')
	axs[0,0].set_title('')
	axs[0,0].set_ylim([1,max(failures_node_affected_soft)+500])
	axs[0,0].set_xticklabels(['1','2','3','4','>4'])
	axs[0,0].set_xticks([1.15,2.15,3.15,4.15,5.15])
	
	axs[0,0].legend(edgecolor="black")
	
	plt.savefig("PLOT_count_nodes_afected_by_failure_" + year_text +".pdf")
	plt.savefig("PLOT_count_nodes_afected_by_failure_" + year_text +".png")
	print("\nPlot in file: <PLOT_count_nodes_afected_by_failure_"+ year_text +".pdf>")
	
	#############################################################################################
	#PLOT TREEMAP
	print("Treemap plot is comment because it need squarify")
	# plt.clf()
	# fig = plt.figure(figsize=(8,8))
	
	
	# labels_data = ["GPU DBE\n"+str(hardware_failures_clasification["GPU"]["GPU DBE"]),"GPU DPR\n"+str(hardware_failures_clasification["GPU"]["GPU DPR"]),"Machine Check Exception\n"+str(hardware_failures_clasification["Processor"]["Machine Check Exception"]),"GPU BUS\n"+str(software_failures_clasification["GPU"]["GPU BUS"]),"GPU XID\n"+str(software_failures_clasification["GPU"]["GPU XID"]),"Other Failures\n"+str(software_failures_clasification["Memory"]["Kernel Panic"]+software_failures_clasification["Memory"]["RDMA Failure"]+software_failures_clasification["Memory"]["LBUG"]+hardware_failures_clasification["Blade"]["Voltage Fault"]+hardware_failures_clasification["Blade"]["Module Failed"])]
	# sizes_data = [hardware_failures_clasification["GPU"]["GPU DBE"], hardware_failures_clasification["GPU"]["GPU DPR"],hardware_failures_clasification["Processor"]["Machine Check Exception"],software_failures_clasification["GPU"]["GPU BUS"],software_failures_clasification["GPU"]["GPU XID"],software_failures_clasification["Memory"]["Kernel Panic"]+software_failures_clasification["Memory"]["RDMA Failure"]+software_failures_clasification["Memory"]["LBUG"]+hardware_failures_clasification["Blade"]["Voltage Fault"]+hardware_failures_clasification["Blade"]["Module Failed"]]

	# c = [(230/255, 16/255, 16/255),(1, 173/255, 153/255),(1, 92/255, 51/255)]
	# c2 = [(102/255, 179/255, 1),(51/255, 51/255, 1)]
	# c3 = [(1, 1, 0)]
	
	# c = c + c2 + c3
	
	# squarify.plot(label=labels_data,sizes=sizes_data, color = c, alpha=.6)
	# plt.title("Failures",fontsize=12,fontweight="bold")
	
	# h1 = mpatches.Patch(color=(237/255, 16/255, 16/255))
	# h2 = mpatches.Patch(color=(1, 173/255, 153/255), label='Hardware')
	# h3 = mpatches.Patch(color=(1, 92/255, 51/255))
	# l = mpatches.Patch(color='white',label='-------------')
	# #s1 = mpatches.Patch(color=)
	# s2 = mpatches.Patch(color=(102/255, 179/255, 1), label='Software')
	# s3 = mpatches.Patch(color=(51/255, 51/255, 1))
	# plt.legend(handles=[h1,h2,h3,l,s2,s3],loc=9, ncol=1,bbox_to_anchor=(1.15, 0.5),edgecolor="black")

	# # #Remove our axes and display the plot
	# plt.axis('off')
	# fig.tight_layout(pad=9)
	
	# plt.savefig("PLOT_Treemap_" + year_text +".pdf")
	# print("\nPlot in file: <PLOT_Treemap_"+ year_text +".pdf>")
	
	# ##############################################################################
	return 
	
	
if len(sys.argv) >= 1:
	dirName = sys.argv[1]
	outputFileName = "" #sys.argv[2]
	f_cabinets(dirName)
else:
	print ("ERROR, usage: %s <directory>" % sys.argv[0])
	sys.exit(0)