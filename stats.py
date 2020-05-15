#!/usr/bin/env python

# Costa Rica High Technology Center (CeNAT)
# Advanced Computing Laboratory
# Esteban Meneses, PhD (esteban.meneses@acm.org)
# Generates a distribution of unique identifiers in a column of a file

import sys
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


### FUNCTIONS ###

def readFile(fileName, column):
	""" Reads a file and returns a hash table with a histogram of unique phrases
on a particular column of a tab-delimited file """
	table = {}
	title = ""
	count = 0

	#size of failure file
	with open(fileName) as f:
		lines = len(f.readlines())

	with open(fileName) as f:
		for line in f:
			fields = line.split('|')
			count += 1
			print ("Progress: %d%%"% (count/lines*100),end="\r")
			sys.stdout.flush()
			try:
				term = fields[column].strip()
				if count == 1:
					title = term +"_"+ fileName[-8:-4]
					continue
				try:
					table[term] += 1
				except KeyError:
					table[term] = 1
			except IndexError:
				print('Unrecognized format in line %d: %s' % (count,line))
	return (table,title)

### MAIN CODE ###
if len(sys.argv) < 3:
	print ("ERROR, usage: %s <file> <column>" % sys.argv[0])
	print ("<file>: failure log file\n<column>: number of column to use for the stats")
	sys.exit(1)

# generating statistics from file
fileName = sys.argv[1]
column = int(sys.argv[2]) - 1
(table,title) = readFile(fileName,column)

# getting values for charts
labels = tuple(table.keys())
sizes = []
for key in table.keys():
	sizes.append(table[key])
gridSize = (2, 1)
fileName = title + '.pdf'

# generating pie chart
print("Generating pie chart...")
plt.subplot2grid(gridSize, (0, 0), rowspan=1, colspan=1)
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title(title.upper(),bbox={'facecolor':'0.8', 'pad':5})
plt.axis('equal')

# generating histogram
print("Generating histogram...")

plt.subplot2grid(gridSize, (1, 0), rowspan=1, colspan=1)
position = np.arange(len(labels))
count = np.array(sizes)
plt.barh(position, count, align='center', alpha=0.4)
plt.yticks(position, labels)
plt.xlabel('Count')
plt.savefig(fileName)

