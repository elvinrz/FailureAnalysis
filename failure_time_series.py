
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
from pandas import Series
#from statsmodels.tsa.seasonal import seasonal_decompose

def movingaverage(values, window):
	weights = np.repeat(1.0, window)/window
	result = np.convolve(values, weights, 'valid')
	return result

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

def time_series(dir_name, diroutput):
	#""" Reads a failure log file and correlates job IDs with MOAB log files in the directory """
	dayFormat = '%a_%b_%d_%Y'
	file_count = 0
	line_count = 0
	pathFileName = []

	bank4 = MCE = bank0 = bank2 = bank6 = DBE = DPR = BUS = XID = SXMP = SXMW = 0


	event_week_GPU_2014  = {}
	event_week_GPU_2015  = {}
	event_week_GPU_2016  = {}
	event_week_GPU_2017  = {}
	event_week_GPU_2018  = {}

	event_week_CPU_2014  = {}
	event_week_CPU_2015  = {}
	event_week_CPU_2016  = {}
	event_week_CPU_2017  = {}
	event_week_CPU_2018  = {}

	event_week_MEMORY_2014  = {}
	event_week_MEMORY_2015  = {}
	event_week_MEMORY_2016  = {}
	event_week_MEMORY_2017  = {}
	event_week_MEMORY_2018  = {}


	event_day = {}
	event_week_2014 = {}
	event_week_2015 = {}
	event_week_2016 = {}
	event_week_2017 = {}
	event_week_2018 = {}

	event_week_CPU_CPU_MEM_total_2014 = {}
	event_week_CPU_CPU_MEM_total_2015 = {}
	event_week_CPU_CPU_MEM_total_2016 = {}
	event_week_CPU_CPU_MEM_total_2017 = {}
	event_week_CPU_CPU_MEM_total_2018 = {}

	#list init
	for i in range (1, 53):
		event_week_2014[i] =  0
		event_week_2015[i] =  0
		event_week_2016[i] =  0
		event_week_2017[i] =  0
		event_week_2018[i] =  0


		event_week_CPU_2014[i] = event_week_CPU_2015[i] = event_week_CPU_2016[i] = event_week_CPU_2017[i] = event_week_CPU_2018[i] = 0
		event_week_GPU_2014[i] = event_week_GPU_2015[i] = event_week_GPU_2016[i] = event_week_GPU_2017[i] = event_week_GPU_2018[i] = 0
		event_week_MEMORY_2014[i] = event_week_MEMORY_2015[i] = event_week_MEMORY_2016[i] = event_week_MEMORY_2017[i] = event_week_MEMORY_2018[i] = 0

		event_week_CPU_CPU_MEM_total_2014[i] = 0
		event_week_CPU_CPU_MEM_total_2015[i] = 0
		event_week_CPU_CPU_MEM_total_2016[i] = 0
		event_week_CPU_CPU_MEM_total_2017[i] = 0
		event_week_CPU_CPU_MEM_total_2018[i] = 0


	init = False
	# start timer
	startTime = time.clock()
    #get all files of the year
	for path, dirs, files in os.walk(dir_name):
			for f in files:
				pathFileName.append(f)


	init_tables(event_day)

	c = 0
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
				description = item[6].strip()
				text = item[7].strip()
				#print(description)

				#initialize all key dates of a range to avoid null dates
				# if init == False:
					# event_day = full_dates(2015,2016, event_day).copy()
					# init = True


				# if d in event_day.keys():
					# event_day[d] += 1


				if description == "Machine Check Exception":
					if "Bank 4" in text:
						bank4 += 1
						c += 1
					if "MCE" in text:
						c += 1
						MCE += 1
					if "Bank 0" in text:
						c += 1
						bank0 += 1
					if "Bank 2" in text:
						c += 1
						bank2 += 1
					if "Bank 6" in text:
						c += 1
						bank6 += 1
				if description == "GPU DBE":
					c += 1
					DBE += 1
				if description == "GPU DPR":
					c += 1
					DPR += 1
				if description == "SXM Warm Temp":
					SXMW += 1
				if description == "GPU BUS":
					c += 1
					BUS += 1
				if description == "GPU XID":
					c += 1
					XID += 1
				if description == "SXM Power Off":
					c += 1
					SXMP += 1


				if week in event_week_2014.keys():
					if year == "2014":
						event_week_2014[week] += 1
						if description == "Machine Check Exception":
							if "Bank 4" in text or "MCE" in text:
								event_week_MEMORY_2014[week] += 1
								event_week_CPU_CPU_MEM_total_2014[week] += 1
							if "Bank 0" in text or "Bank 2" in text or "Bank 6" in text:
								event_week_CPU_2014[week] += 1
								event_week_CPU_CPU_MEM_total_2014[week] += 1
						if description == "GPU DBE" or description == "GPU DPR" or description == "GPU BUS" or description == "GPU XID" or description == "SXM Power Off" or description == "SXM Warm Temp":
							event_week_GPU_2014[week] += 1
							event_week_CPU_CPU_MEM_total_2014[week] += 1
						continue



				if week in event_week_2015.keys():
					if year == "2015":
						event_week_2015[week] += 1
						if description == "Machine Check Exception":
							if "Bank 4" in text or "MCE" in text:
								event_week_MEMORY_2015[week] += 1
								event_week_CPU_CPU_MEM_total_2015[week] += 1
							if "Bank 0" in text or "Bank 2" in text or "Bank 6" in text:
								event_week_CPU_2015[week] += 1
								event_week_CPU_CPU_MEM_total_2015[week] += 1
						if description == "GPU DBE" or description == "GPU DPR" or description == "GPU BUS" or description == "GPU XID" or description == "SXM Power Off" or description == "SXM Warm Temp":
							event_week_GPU_2015[week] += 1
							event_week_CPU_CPU_MEM_total_2015[week] += 1
						continue

				if week in event_week_2016.keys():
					if year == "2016":
						event_week_2016[week] += 1
						if description == "Machine Check Exception":
							if "Bank 4" in text or "MCE" in text:
								event_week_MEMORY_2016[week] += 1
								event_week_CPU_CPU_MEM_total_2016[week] += 1
							if "Bank 0" in text or "Bank 2" in text or "Bank 6" in text:
								event_week_CPU_2016[week] += 1
								event_week_CPU_CPU_MEM_total_2016[week] += 1
						if description == "GPU DBE" or description == "GPU DPR" or description == "GPU BUS" or description == "GPU XID" or description == "SXM Power Off" or description == "SXM Warm Temp":
							event_week_GPU_2016[week] += 1
							event_week_CPU_CPU_MEM_total_2016[week] += 1
						continue


				if week in event_week_2017.keys():
					if year == "2017":
						event_week_2017[week] += 1
						if description == "Machine Check Exception":
							if "Bank 4" in text or "MCE" in text:
								event_week_MEMORY_2017[week] += 1
								event_week_CPU_CPU_MEM_total_2017[week] += 1
							if "Bank 0" in text or "Bank 2" in text or "Bank 6" in text:
								event_week_CPU_2017[week] += 1
								event_week_CPU_CPU_MEM_total_2017[week] += 1
						if description == "GPU DBE" or description == "GPU DPR" or description == "GPU BUS" or description == "GPU XID" or description == "SXM Power Off" or description == "SXM Warm Temp":
							event_week_GPU_2017[week] += 1
							event_week_CPU_CPU_MEM_total_2017[week] += 1
						continue

				if week in event_week_2018.keys():
					if year == "2018":
						event_week_2018[week] += 1
						if description == "Machine Check Exception":
							if "Bank 4" in text or "MCE" in text:
								event_week_MEMORY_2018[week] += 1
								event_week_CPU_CPU_MEM_total_2018[week] += 1
							if "Bank 0" in text or "Bank 2" in text or "Bank 6" in text:
								event_week_CPU_2018[week] += 1
								event_week_CPU_CPU_MEM_total_2018[week] += 1
						if description == "GPU DBE" or description == "GPU DPR" or description == "GPU BUS" or description == "GPU XID" or description == "SXM Power Off" or description == "SXM Warm Temp":
							event_week_GPU_2018[week] += 1
							event_week_CPU_CPU_MEM_total_2018[week] += 1
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

	#print("Total 2014 CPU: "+ str(event_week_CPU_2014.values()))

	#print("bank2: "+str(bank2))
	#print("bank6: "+str(bank6))
	print("DBE: "+str(DBE))
	print("XID: "+str(XID))
	print("BUS: "+str(BUS))
	print("DPR: "+str(DPR))
	print("SXM power: "+str(SXMP))
	print("bank0, 2, 6: "+str(bank0+bank2+bank6))
	print("bank4: "+str(bank4))
	print("MCE: "+str(MCE))
	print("total:"+str(c))



	#print("SXM warm: "+str(SXMW))

	r = 0
	for i in event_week_GPU_2014.values():
		r = r + i
	for i in event_week_GPU_2015.values():
		r = r + i
	for i in event_week_GPU_2016.values():
		r = r + i
	for i in event_week_GPU_2017.values():
		r = r + i
	for i in event_week_GPU_2018.values():
		r = r + i

	print("total all period GPU:"+str(r))

	r = 0
	for i in event_week_CPU_2014.values():
		r = r + i
	for i in event_week_CPU_2015.values():
		r = r + i
	for i in event_week_CPU_2016.values():
		r = r + i
	for i in event_week_CPU_2017.values():
		r = r + i
	for i in event_week_CPU_2018.values():
		r = r + i

	print("total all period CPU:"+str(r))



	r = 0
	for i in event_week_MEMORY_2014.values():
		r = r + i
	for i in event_week_MEMORY_2015.values():
		r = r + i
	for i in event_week_MEMORY_2016.values():
		r = r + i
	for i in event_week_MEMORY_2017.values():
		r = r + i
	for i in event_week_MEMORY_2018.values():
		r = r + i

	print("total all period MEMORY:"+str(r))

	#print("Total 2014 MEMORY: "+ str(event_week_MEMORY_2014.values()))

	#print("Total 2015 CPU: "+ str(event_week_CPU_2015.values()))
	#print("Total 2015 GPU: "+ str(event_week_GPU_2015.values()))
	#print("Total 2015 MEMORY: "+ str(event_week_MEMORY_2015.values()))

	#print("Total 2016 CPU: "+ str(event_week_CPU_2016.values()))
	#print("Total 2016 GPU: "+ str(event_week_GPU_2016.values()))
	#print("Total 2016 MEMORY: "+ str(event_week_MEMORY_2016.values()))

	#print("Total 2017 CPU: "+ str(event_week_CPU_2017.values()))
	#print("Total 2017 GPU: "+ str(event_week_GPU_2017.values()))
	#print("Total 2017 MEMORY: "+ str(event_week_MEMORY_2017.values()))

	#print("Total 2018 CPU: "+ str(event_week_CPU_2018.values()))
	#print("Total 2018 GPU: "+ str(event_week_GPU_2018.values()))
	#print("Total 2018 MEMORY: "+ str(event_week_MEMORY_2018.values()))

	# print("\nPrcessing 1 plot of 2")
	# plt.clf()
	# #plt.style.use('seaborn-whitegrid')
	# plt.xlabel('Weeks')
	# plt.ylabel('Count of failures')
	# plt.title('Failures by week 2014-2018 ')
	# ax = plt.gca()
	# ax.tick_params(axis = 'x', which = 'major', labelsize = 6)
	# plt.xticks(np.arange(0, 54, 2))
	# plt.plot(range(len(event_week_2014)),list(event_week_2014.values()), 'g--', linewidth=0.5, label="2014 failures")
	# plt.plot(range(len(event_week_2015)),list(event_week_2015.values()), 'g-', linewidth=0.5, label="2015 failures")
	# plt.plot(range(len(event_week_2016)),list(event_week_2016.values()), 'r-', linewidth=1, label="2016 failures")
	# plt.plot(range(len(event_week_2017)),list(event_week_2017.values()), 'm-', linewidth=0.5, label="2017 failures")
	# plt.plot(range(len(event_week_2018)),list(event_week_2018.values()), 'c-', linewidth=0.5, label="2018 failures")

	# plt.legend(framealpha=1,shadow=True, borderpad = 1, fancybox=True)
	# plt.savefig(diroutput+"PLOT_week_2014_2018.pdf")
	# print("\nPlot in file: <PLOT_week_2014_2018.pdf>")


	items = list(event_week_CPU_CPU_MEM_total_2014.values())+list(event_week_CPU_CPU_MEM_total_2015.values())+list(event_week_CPU_CPU_MEM_total_2016.values())+list(event_week_CPU_CPU_MEM_total_2017.values())+list(event_week_CPU_CPU_MEM_total_2018.values())
	#print("todos los valores")
	print(items)
	#print(len(items))
	r = movingaverage(items,8)
	r = np.append(r, r[len(r)-1])
	r = np.append(r, r[len(r)-1])

	r = np.append(r, r[len(r)-1])
	r = np.append(r, r[len(r)-1])
	r = np.append(r, r[len(r)-1])
	r = np.append(r, r[len(r)-1])
	r = np.append(r, r[len(r)-1])




	#print(r)
	#print(len(r))

	print("\nProcessing plot ...")

	plt.clf()
	plt.figure(figsize=(8,2.5	))
	plt.xlabel('Weeks')
	ax = plt.gca()
	ax.tick_params(axis = 'x', which = 'major', labelsize = 6.5)
	plt.ylabel('Failure Count')
	plt.title('Failures by week')
	plt.xticks([1,8,16,24,32,40,48,52,60,68,76,84,92,100,104,112,120,128,136,144,152,156,164,172,180,188,196,204,208,216,222,230,238,246,254,258],["1","8","16","24","32","40","48","1","8","16","24","32","40","48","1","8","16","24","32","40","48","1","8","16","24","32","40","48","1","8","16","24","32","40","48",""])

	plt.plot(range(len(event_week_2014)+len(event_week_2015)+len(event_week_2016)+len(event_week_2017)+len(event_week_2018)),list(event_week_2014.values())+list(event_week_2015.values())+list(event_week_2016.values())+list(event_week_2017.values())+list(event_week_2018.values()), 'k-', linewidth=0.3, label="Total System Failures")

	#plt.plot(range(len(event_week_CPU_CPU_MEM_total_2014)+len(event_week_CPU_CPU_MEM_total_2015)+len(event_week_CPU_CPU_MEM_total_2016)+len(event_week_CPU_CPU_MEM_total_2017)+len(event_week_CPU_CPU_MEM_total_2018)),list(event_week_CPU_CPU_MEM_total_2014.values())+list(event_week_CPU_CPU_MEM_total_2015.values())+list(event_week_CPU_CPU_MEM_total_2016.values())+list(event_week_CPU_CPU_MEM_total_2017.values())+list(event_week_CPU_CPU_MEM_total_2018.values()), 'k--', linewidth=0.1, label="Total Failures")

	#print(list(event_week_2014.values())+list(event_week_2015.values())+list(event_week_2016.values())+list(event_week_2017.values())+list(event_week_2018.values()))

	plt.plot(range(len(event_week_GPU_2014)+len(event_week_GPU_2015)+len(event_week_GPU_2016)+len(event_week_GPU_2017)+len(event_week_GPU_2018)),list(event_week_GPU_2014.values())+list(event_week_GPU_2015.values())+list(event_week_GPU_2016.values())+list(event_week_GPU_2017.values())+list(event_week_GPU_2018.values()), 'r-', linewidth=0.4, label="GPU")

	plt.plot(range(len(event_week_CPU_2014)+len(event_week_CPU_2015)+len(event_week_CPU_2016)+len(event_week_CPU_2017)+len(event_week_CPU_2018)),list(event_week_CPU_2014.values())+list(event_week_CPU_2015.values())+list(event_week_CPU_2016.values())+list(event_week_CPU_2017.values())+list(event_week_CPU_2018.values()), 'g-', linewidth=0.4, label="CPU")

	plt.plot(range(len(event_week_MEMORY_2014)+len(event_week_MEMORY_2015)+len(event_week_MEMORY_2016)+len(event_week_MEMORY_2017)+len(event_week_MEMORY_2018)),list(event_week_MEMORY_2014.values())+list(event_week_MEMORY_2015.values())+list(event_week_MEMORY_2016.values())+list(event_week_MEMORY_2017.values())+list(event_week_MEMORY_2018.values()), 'b-', linewidth=0.4, label="Memory")

	#plt.plot(range(len(event_week_GPU_2014)+len(event_week_GPU_2015)+len(event_week_GPU_2016)+len(event_week_GPU_2017)+len(event_week_GPU_2018)),r, 'b-', linewidth=0.7, label="GPU MovAvg")

	#plt.plot(range(len(event_week_GPU_2014)+len(event_week_GPU_2015)+len(event_week_GPU_2016)+len(event_week_GPU_2017)+len(event_week_GPU_2018)),list(event_week_CPU_CPU_MEM_total_2014.values())+list(event_week_CPU_CPU_MEM_total_2015.values())+list(event_week_CPU_CPU_MEM_total_2016.values())+list(event_week_CPU_CPU_MEM_total_2017.values())+list(event_week_CPU_CPU_MEM_total_2018.values()), 'b-', linewidth=0.7, label="GPU MovAvg")



	#plt.legend(framealpha=0.5,shadow=False, borderpad = 1, fancybox=False,prop={'size': 6})
	plt.axvline(x=52, color='k', linestyle='dashed', linewidth=0.3)
	plt.axvline(x=104, color='k', linestyle='dashed', linewidth=0.3)
	plt.axvline(x=156, color='k', linestyle='dashed', linewidth=0.3)
	plt.axvline(x=208, color='k', linestyle='dashed', linewidth=0.3)

	#plt.axvline(x=109, color='b', linestyle=(0, (3, 5, 1, 5, 1, 5)), linewidth=0.7)
	#plt.axvline(x=136, color='b', linestyle='dashed', linewidth=1)
	#plt.axvline(x=176, color='b', linestyle='dashed', linewidth=1)


	plt.text(10 ,120,"2014")
	plt.text(68 ,120,"2015")
	plt.text(120 ,120,"2016")
	plt.text(173 ,120,"2017")
	plt.text(212 ,120,"2018")

	plt.legend(edgecolor="black",prop={'size': 6},loc=0)

	plt.tight_layout()
	plt.savefig(diroutput+"PLOT_all_years_week.pdf")
	print("\nPlot in file: <PLOT_all_years_week.pdf>")


	#series = Series.from_csv('failure_week.csv', header=0)
	#print(series)
	#result = seasonal_decompose(series, model='multiplicative', freq=4)
	#fig, axes = plt.subplots(4, 1, sharex=True)

	#result.observed.plot(ax=axes[0], legend=False, color='r', linewidth=0.5)
	#axes[0].set_ylabel('Observed')
	#result.trend.plot(ax=axes[1], legend=False, color='g', linewidth=0.5)
	#axes[1].set_ylabel('Trend')
	#result.seasonal.plot(ax=axes[2], legend=False, linewidth=0.5)
	#axes[2].set_ylabel('Seasonal')
	#result.resid.plot(ax=axes[3], legend=False, color='k', linewidth=0.5)
	#axes[3].set_ylabel('Residual')

	#plt.savefig(diroutput+"test.pdf")

	return


if len(sys.argv) >= 3:
	dirName = sys.argv[1]
	diroutput = sys.argv[2]
	outputFileName = "" #sys.argv[2]
	time_series(dirName,diroutput)
else:
	print ("ERROR, usage: %s <directory> <output dir>" % sys.argv[0])
	sys.exit(0)
