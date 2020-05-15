#!/usr/bin/env python

# Costa Rica High Technology Center (CeNAT)
# Advanced Computing Laboratory
# Elvis Rojas, MSc (erojas@una.cr)
# Esteban Meneses, PhD (esteban.meneses@acm.org)
# Extracts mean-time-between-failures (MTBF) from the failure log, optionally filtered by the value of a field

import sys
import re
import datetime
import matplotlib as mtl
mtl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

### FUNCTIONS ###

def readFile(fileName,outputFailureFile, delta, column=0, value=0):
	""" Reads a file and returns a list with the MTBF """
	mtbf = []
	formatAlt = '%m/%d/%y %H:%M %p'
	date_format = '%Y-%m-%d %H:%M:%S'
	previousDate = 0
	count = 0
	
	with open(fileName) as f:
		lines = len(f.readlines())
	
	file = open(outputFailureFile, 'w')
	with open(fileName) as f:
		for line in f:
			if count == 0:
			    file.write(line)
			count += 1
			
			print ("Progress: %d%%" % (count/lines*100), end = "\r")
			sys.stdout.flush()	
			
			if count == 1:
				continue
			fields = line.split('|')
			dateAndTime = fields[3].strip()
			if column != 0 and fields[column] != value:
				continue
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
				print(diff_seconds)
				exit(1)
			#if diff_seconds < delta:
				#continue
			mtbf.append(diff_seconds)
			file.write(line)
			previousDate = currentDate
		file.close()			
	return mtbf

### MAIN CODE ###
#DEBUG print 'Number of arguments:', len(sys.argv), 'arguments.'
#DEBUG print 'Argument List:', str(sys.argv)

if len(sys.argv) == 6:
	fileName = sys.argv[1]
	outputFile = sys.argv[2]
	delta = int(sys.argv[3])
	bins = int(sys.argv[4])
	outputFailureFile = sys.argv[5]
	mtbf = readFile(fileName,outputFailureFile, delta)
elif len(sys.argv) == 8:
	fileName = sys.argv[1]
	outputFile = sys.argv[2]
	delta = int(sys.argv[3])
	bins = int(sys.argv[4])
	outputFailureFile = sys.argv[5]
	column = int(sys.argv[6]) - 1
	value = sys.argv[7]
	mtbf = readFile(fileName,outputFailureFile,delta,column,value)
else:
	print("ERROR, usage: %s <input file> <output file> <delta> <bins> <output file> [<column> <value>] " % sys.argv[0])
	print("<input file>: failure log file")
	print("<output file>: PDF file name for figure")
	print("<delta>: minimum difference in seconds between failures")
	print("<bins>: number of bins in the histogram")
	print("<output file>: output text file for filtered failures")
	print("<column>: field in the failure log file to filter records")
	print("<value>: value of the field for filtering")
	sys.exit(0)

# validation, the mtbf values should add up to 1 year or less
if sum(mtbf) > 31536000:
	print("WARNING: MTBF values add up to %d seconds, more than one year (31536000 seconds)" % sum(mtbf))

# plotting MTBF
plt.title('Mean Time Between Failures (MTBF)')
plt.hist(mtbf, bins)
plt.xlabel('Seconds')
plt.ylabel('Count')
#plt.axis([0,75000, 0,5000])
plt.savefig(outputFile)
#plt.show()
