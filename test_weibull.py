
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
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from scipy.stats import weibull_min
from scipy.stats import gamma
import random

def get_MTBF_of_array_dates(failure_list):
	previousDate = 0
	mtbf_result = []
	format = '%Y-%m-%d %H:%M:%S'
	
	for dateAndTime in failure_list:
		dateAndTime = dateAndTime.strip()
		try:
			currentDate = datetime.datetime.strptime(dateAndTime, format)
		except ValueError:
			print("ERROR with date %s" % (dateAndTime))
			sys.exit(0)
				
		if previousDate == 0:
			previousDate = currentDate
			continue
		
		diff = currentDate - previousDate
		
		diff_seconds = diff.days*24*60*60 + diff.seconds
		if diff_seconds < 0:
			print("ERROR: negative event time: "+str(currentDate) +"/"+str(previousDate)  )
			sys.exit()
		#file.write(str(diff_seconds)+"\n")
		mtbf_result.append(diff_seconds)
		previousDate = currentDate

	return mtbf_result

def only_cdf_plot(dirName, output_dir):
	mtbf_resambled = []
	pathFileName_resampled = []
	file_name = ""
	count = 0
	for path, dirs, files in os.walk(dirName):
			for f in files:
				pathFileName_resampled.append(f)
				
	
	for file_name in pathFileName_resampled:  
		f = file_name
		file_name = dirName + file_name
		print("Using resampled file: "+file_name)	
		with open(file_name) as log:
			for line in log:
				print(line)
				mtbf_resambled.append(int(line.strip()))
				count += 1
	
		print("Count data processing: "+str(count))
		plot_distributions(mtbf_resambled,f,dirName)
			
def plot_distributions(data,label,output_dir):
	
	d = []
	for i in data:
		#if int(i) != 0: # and int(i) < 280000:
		d.append(i)

	x = np.linspace(min(data), max(data), len(data))#np.sort(data)
	mu = np.mean(x)
	sigma = np.std(x)
	n_bins = 30

	fig, ax = plt.subplots(figsize=(4, 2.5))#(4, 3))

	data = data1 = np.sort(d)
	mu = np.mean(data, keepdims = True)
	sigma = np.std(data)

	plt.xscale('log')
	################################################
	#Data plot
	y = np.arange(1, len(data)+1)/len(data)
	plt.plot(data, y, 'r-', linewidth=0.9, label="Data")

	################################################
	#Exponential plot
	loc,scale=ss.expon.fit(data, floc=0)

	y = ss.expon.cdf(data, loc, scale)

	D, P = ss.kstest(data, lambda x : y)

	plt.plot(data, y,'m-.', linewidth=0.6, label="Exponential - KS D="+str(round(D, 3)))
	print("Exponential KS D Value: " + str(D) + " - P value: " + str(P))

	################################################

	#lognormal plot
	logdata = np.log(data)
	#estimated_mu, estimated_sigma, scale = ss.norm.fit(logdata)
	shape, loc, scale = ss.lognorm.fit(data,floc=0)

	#scale = estimated_mu
	#s = estimated_sigma 
	#y = (1+scipy.special.erf((np.log(data)-scale)/(np.sqrt(2)*s)))/2 #ss.lognorm.cdf(data, s, scale) 
	y  = ss.lognorm.cdf(data, shape, loc, scale) 

	D, P = ss.kstest(data, lambda x : y)

	plt.plot(data, y, 'c:', linewidth=0.6, label="Lognormal - KS D="+str(round(D, 3))) 
	print("Lognormal KS D Value: " + str(D) + " - P value: " + str(P) )
	#################################################
	#Weibull

	shape, loc, scale = ss.weibull_min.fit(data, floc=0)
	
	print("shape")
	print(shape)
	
	wei = ss.weibull_min(shape, loc, scale) # shape, loc, scale - creates weibull object
	#x = np.linspace(np.min(data), np.max(data), len(data))
	wei = ss.weibull_min(shape, loc, scale)

	meanw, var = weibull_min.stats(shape,loc,scale, moments='mv')

	D, P = ss.kstest(data, lambda x : wei.cdf(data))

	plt.plot(data, wei.cdf(data),'b-',linewidth=0.6, label="Weibull - KS D="+str(round(D, 3)))
	
	#################################################
	#Gamma
	shape1, loc1, scale1 = gamma.fit(data, floc=0)
	
	y = gamma.cdf(x=data, a=shape1, loc=loc1, scale=scale1)
	
	D, P = ss.kstest(data, lambda x : y)
	
	plt.plot(data, y,'g--',linewidth=0.6, label="Gamma - KS D="+str(round(D, 3)))
	
	
	plt.legend(edgecolor="black",prop={'size': 7})


	print("Weibull KS D Value: " + str(D) + " - P value: " + str(P) )
	print("---------------------------------------------------------------------------------------")
	
	print("Weibull Mean: "+str(meanw/60/60))
	print("Weibull Var: "+str(var/60/60))
	# print(data)
	# print(wei.cdf(data))
	#################################################
	plt.xlabel('TBF (seconds)')
	plt.ylabel('Cumulative Probability')
	#plt.legend(framealpha=1,shadow=True, borderpad = 1, fancybox=True)
	plt.tight_layout()
	plt.savefig(output_dir+"plot_cdf_"+label+".pdf")

def processing_weibull(output_dir):
	pathFileName = []
	cpu_failures = []
	gpu_failures = []
	memory_failures = []
	
	c = []
	t = []
	
	
	
		
	# mtbf_cpu = get_MTBF_of_array_dates(cpu_failures)
	# mtbf_gpu = get_MTBF_of_array_dates(gpu_failures)
	# mtbf_memory = get_MTBF_of_array_dates(memory_failures)
	
	s = 0.5
	a = np.random.weibull(s, 3)
	b = np.random.weibull(s, 3)
	
	print("a original")
	print(a)
	print("b original")
	print(b)
	
	plot_distributions(a,"wei",output_dir)
	plot_distributions(b,"wei2",output_dir)
	
	for i in range(len(a)-1):
		c.append(0)
		c[i] = a[i]+a[i+1]
		a[i+1] = c[i]
	
	for i in range(len(b)-1):
		c.append(0)
		c[i] = b[i]+b[i+1]
		b[i+1] = c[i]
		
	for i in a:
		t.append(i)
	for i in b:
		t.append(i)

	t.sort()
	
	print("a procesado")
	print(a)        
	print("b procesado")
	print(b)
	print("t unido")
	print(t)
	
	c.append(0)
	c[0] = t[0]
	for i in range(len(t)-1):
		c.append(0)
		c[i+1] = t[i+1]-t[i]
	
	while 0 in c: c.remove(0)
	
	print("result")
	print(c)
	
	plot_distributions(c,"wei_total",output_dir)
	
	
if len(sys.argv) >= 2:
	output_dir = sys.argv[1]
	processing_weibull(output_dir)