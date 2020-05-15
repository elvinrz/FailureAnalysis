#!/usr/bin/env python

# Costa Rica High Technology Center (CeNAT)
# Advanced Computing Laboratory
# Elvis Rojas, MSc (erojas@una.ac.cr)
# Esteban Meneses, PhD (esteban.meneses@acm.org)
# Extracts mean-time-between-failures (MTBF) from the failure log, optionally filtered by the value of a field

import sys
import re
import datetime
import time
import matplotlib as mtl
mtl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from scipy.stats import norm
import scipy.special
from collections import Counter

import statsmodels.api as sm



### FUNCTIONS ###

def readFile(fileName,outputFile):
	""" Reads a file and returns a list with the MTBF """
	mtbf = []
	countList = []
	formatAlt = '%m/%d/%y %H:%M %p'
	date_format = "%Y-%m-%d %H:%M:%S"
	previousDate = 0
	count = 0
	
	outputFile_data = outputFile[:-4]
	outputFile_data = outputFile_data + ".txt"
	
	file = open(outputFile_data, 'w')
	
	with open(fileName) as f:
		#f.seek(0)
		for line in f:
			if count == 0:
			    file.write(line)
			count += 1

			if count == 1:
				continue
			fields = line.split('|')
			#print("++++++++++++++++++++++=%s - %d"%(fields[3].strip(),count))
			
			dateAndTime = fields[3].strip()
			# print("contador: "+str(count))
			# print(fields)
			# print("linea: " + line)
			# print("Fecha: " + dateAndTime)
			
			try:
				currentDate = datetime.datetime.strptime(dateAndTime, date_format)
			except ValueError:
				try:
					currentDate = datetime.datetime.strptime(dateAndTime, formatAlt)
				except ValueError:
					print("ERROR on line %d with date %s" % (count,dateAndTime))
					sys.exit(0)
					
			if previousDate == 0:
				previousDate = currentDate
				continue
			
			diff = currentDate - previousDate
			diff_seconds = diff.days*24*60*60 + diff.seconds
			
			if diff_seconds < 0:
				print("ERROR: negative event time difference in line %d" % count)
				sys.exit()
			
			#print("Dif seconds: %d count: %d "% (diff_seconds, count))
			mtbf.append(diff_seconds)	
			
			countList.append(count)
			file.write(str(diff_seconds)+"\n")
			previousDate = currentDate
		file.close()
		mean = np.mean(mtbf)
		print(mean/60/60)
		
		#Plot MTBF distribution
		
		plt.figure(figsize=(3,3))
		h = plt.hist(mtbf, 60, color = "blue", lw=0.1,edgecolor='black')
		plt.tick_params(labelsize=7)
		plt.xlabel('Seconds')
		plt.ylabel('Frequency')
		plt.axvline(x=mean, color='r', linestyle='dashed', linewidth=1)
		plt.text(mean + 10000,h[0][0]-15,str(round((mean/60/60),3))+ " hours")
	
		plt.tight_layout()
		plt.savefig("Plot_mtbf.pdf")
		
	return (mtbf, countList)
	
if len(sys.argv) == 3:
	fileName = sys.argv[1]
	outputFile = sys.argv[2]
	(data, countList) = readFile(fileName,outputFile)
else:
	print("ERROR, usage: %s <input file> <output file>" % sys.argv[0])
	print("<input file>: failure log file")
	print("<output file>: PDF file name for plot ")
	sys.exit(0)

#np.random.seed(19680801)

x = np.linspace(min(data), max(data), len(data))#np.sort(data)
mu = np.mean(x)
sigma = np.std(x)
n_bins = 30

fig, ax = plt.subplots(figsize=(4, 3))


#data manipulation for plot
d = []
for i in data:
	if int(i) != 0:
		d.append(i)

		
data = data1 = np.sort(d)
mu = np.mean(data, keepdims = True)
sigma = np.std(data)

plt.xscale('log')
################################################
#Data plot
y = np.arange(1, len(data)+1)/len(data)
plt.plot(data, y, 'g-', linewidth=1.5, label="Data")

################################################
#Exponential plot
loc,scale=ss.expon.fit(data)

y = ss.expon.cdf(data, loc, scale)

D, P = ss.kstest(data, lambda x : y)

plt.plot(data, y,linestyle='-.',color="0.30", linewidth=0.9, label="Exponential - KS D="+str(round(D, 3)))
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

plt.plot(data, y, linestyle =':', color= "0.30", linewidth=1.2, label="Lognormal - KS D="+str(round(D, 3))) 
print("Lognormal KS D Value: " + str(D) + " - P value: " + str(P) )
#################################################
#Weibull

shape, loc, scale = ss.weibull_min.fit(data, floc=0)
wei = ss.weibull_min(shape, loc, scale) # shape, loc, scale - creates weibull object
#x = np.linspace(np.min(data), np.max(data), len(data))

D, P = ss.kstest(data, lambda x : wei.cdf(data))

plt.plot(data, wei.cdf(data),'b--',linewidth=0.9, label="Weibull - KS D="+str(round(D, 3)))
plt.legend(edgecolor="black",prop={'size': 8})


print("Weibull KS D Value: " + str(D) + " - P value: " + str(P) )
print("---------------------------------------------------------------------------------------")

# print(data)
# print(wei.cdf(data))
#################################################
plt.xlabel('Time Between Failures (seconds)')
plt.ylabel('Cumulative Probability')
#plt.legend(framealpha=1,shadow=True, borderpad = 1, fancybox=True)
plt.tight_layout()
plt.savefig(outputFile)
plt.savefig(outputFile[:-4]+".png")

#QQ PLOT

plt.clf()
norm=np.random.normal(0,2,len(data))
norm.sort()
plt.plot(norm,data,"b.")
z = np.polyfit(norm,data, 1)
p = np.poly1d(z)
plt.plot(norm,p(norm),"r--", linewidth=1)
plt.savefig("QQ_plot")


