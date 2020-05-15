
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
import collections
# import scipy.stats as ss
# from collections import defaultdict
# import collections
# from datetime import date, timedelta
# import seaborn as sns
# from datetime import date, timedelta, datetime as dt
# import calendar as cl
# #import squarify
# import matplotlib.patches as mpatches
# import matplotlib.gridspec as gridspec


	
def fr_nodes(file_name, workload_dirName):
	#""" Reads a failure log file and correlates job IDs with MOAB log files in the directory """
	file_count = 0
	line_count = 0
	pathFileName = []
	pathFileName_workload = []
	workload_node_req = []
	data_node_req  = []
	data_node_fail = []
	data_node_req_fail = {}
	count_nodes_failed_nodes = 0 

	year_text = ""
	format = '%Y-%m-%d %H:%M:%S'
	make_file = False
	
	#read the workload filter 
	try:
		file = open("workload_filter.txt", "r") 
		workload_node_req.clear()
		for line in file: 
			workload_node_req.append(line)
		print("Workload file loaded")
	except IOError:
		make_file = True
		print("Workload file created")
	
	#############################################################
	#############################################################
	#extract workload data
	#get all files of the year
	if make_file == True:
		for path, dirs, files in os.walk(workload_dirName):
			for d in dirs:
				for f in glob.iglob(os.path.join(path, d, '*')):
					pathFileName_workload.append(f)
		
		l = len(pathFileName_workload)
		count = 0
		for file_name_workload in pathFileName_workload:	
			print ("Workload stage / Progress: %d%%"% (count/l*100),end="\r")	
			count += 1
			with open(file_name_workload) as log_workload:
				for event_workload in log_workload:
					c = event_workload.split()
					objid = c[3]
					if c[2] != "job" or c[4] != "JOBEND":
						continue
		
					if objid == "0":
						print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
						print("id 2224753: "+c[5] +" - "+str(ceil(int(c[5])/16)) +" - "+str(ceil(int(c[6])/16)))
						print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
					
					columns_check = event_workload.split()
					if "STARTTIME" in event_workload:
						for item in columns_check:
							if  "REQUESTEDTC" in item.strip():
								req_nodes = ceil(int(item[12:])/16)
					else:
						if c[5] == 0:
							req_nodes = int(c[5])
						else: req_nodes = ceil(int(c[6])/16)
					
					#if req_nodes != 0:
					workload_node_req.append(objid + " " + str(req_nodes) + " " + file_name_workload)
			
			
		wl = open("workload_filter.txt", 'w')
		for n in workload_node_req:
			wl.write(n + "\n")
		wl.close()
			
	#############################################################
	#############################################################
	
	
	# start timer
	startTime = time.clock()
	count = 0
	#size of failure file
	with open(file_name) as f:
		l = len(f.readlines())
	
	#list to contein data
	elements = []
	for i in range(l):
		elements.append([])
	
	
	#initializing dictionary to store node req and node fail
	for i in range(l):
		data_node_req_fail[i] = ""
	
	with open(file_name) as log:
		
		if line_count == 1:
			next(log)
		for line in log:
			print ("Filter stage / Progress: %d%%"% (count/l*100),end="\r")	
			item = line.split("|")
			jobid = item[2].strip()
			dateAndTime = item[3].strip()												# reading time
			month = dateAndTime[5:-12] 
			all_nodes = item[9].strip().split()
			count_nodes_failed_nodes = len(all_nodes)
			
			found = False
			for n in workload_node_req:
				i = n.split()
				if i[0].strip() == jobid:
					#data_node_req.append(i[1])
					#data_node_fail.append(count_nodes_failed_nodes)
					#data_node_req_fail[i[1]] = i[1] +"|"+str(count_nodes_failed_nodes) + "|" + i[2]
					elements[count].append(int(i[1]))
					elements[count].append(count_nodes_failed_nodes)
					elements[count].append(month)
					
					#elements[count].append(i[2])
					#elements[count].append(jobid)
					found = True
					break
			if found == False:
				print("ID " + jobid + " does not found in workload dataset / " )
			count += 1
			
	
	#od = collections.OrderedDict(sorted(data_node_req_fail.items()))
	new_elements = []
	for n in elements:
		print(n)
	
	new_elements = filter(None, elements)#elements[1:]
	new_elements = sorted(new_elements,key=lambda l:l[0])
	
	# print("-----------------------")
	# print(new_elements)
	nfail = []
	nreq = []
	node_month = []
	for i in new_elements:
		nfail.append(i[1])
		nreq.append(i[0])
		node_month.append(int(i[2]))
	
	print("-----------------------")
	print(nfail)
	
	print("-----------------------")
	print(nreq)
	
	fig = plt.figure()
	ax = plt.gca()
	ax.plot(nreq,nfail, '.', color='black')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel("Requested Node")
	ax.set_ylabel("Failed Node")
	plt.savefig("PLOT_node_req_node_fail" + ".pdf")
	
	print("\nPLOT_node_req_node_fail" + workload_dirName[5:-1] + ".pdf")
	
	fig = plt.figure()
	ax = plt.gca()
	cm = plt.cm.get_cmap('Paired')
	sc = plt.scatter(nreq, nfail, c=node_month, vmin=1, vmax=12, s=15, cmap=cm, linewidth=0.3, edgecolors="black")
	cb = plt.colorbar(sc)
	cb.set_label("Month")
	
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel("Requested Node")
	ax.set_ylabel("Failed Node")
	plt.savefig("PLOT_node_req_node_fail" + "_2.pdf")
	
	print("\nPLOT_node_req_node_fail" + workload_dirName[5:-1] + ".pdf")
	
	
	return 
	
	
if len(sys.argv) >= 1:
	fileName = sys.argv[1]
	workload_dirName = sys.argv[2]
	fr_nodes(fileName, workload_dirName)
else:
	print ("ERROR, usage: %s <failures dataset directory> <workload directory>" % sys.argv[0])
	sys.exit(0)