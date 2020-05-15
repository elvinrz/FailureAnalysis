
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


def pearson(fileName1, column_name1, separator1,fileName2, column_name2, separator2):
	line_count = 0
	column_count = 0
	list_dataX = []
	list_dataY = []


	with open(fileName1) as log:
		for line in log:
			line_count += 1
			if separator1 == "sp":
				columns = line.split()
			else:
				columns = line.split(separator1)
			if line_count == 1:
				for c in columns:
					if c == column_name1:
						next(log)
						break
					column_count += 1
			else:		
				list_dataX.append(int(columns[column_count]))
					
	line_count = 0
	column_count = 0
	with open(fileName2) as log:
		for line in log:
			line_count += 1
			if separator2 == "sp":
				columns = line.split()
			else:
				columns = line.split(separator2)
			if line_count == 1:
				for c in columns:
					if c == column_name2:
						next(log)
						break
					column_count += 1			
			else:
				list_dataY.append(int(columns[column_count]))
	
	#Correlations
	pearson_correlation = ss.stats.pearsonr(list_dataX, list_dataY)
	spearman_correlation = ss.stats.spearmanr(list_dataX, list_dataY)
	kendall_correlation = ss.stats.kendalltau(list_dataX, list_dataY)
	
	#linear regresion
	slope, intercept, r_value, p_value, std_err = ss.linregress(list_dataX,list_dataY)
	
	print("Data X: ", column_name1)
	print("Data Y: ", column_name2)
	
	print("Pearson correlation")
	print("Result: ", pearson_correlation)
	print("------------------------------")
	print("Spearman correlation")
	print("Result: ", spearman_correlation)
	print("------------------------------")
	print("Kendall correlation")
	print("Result: ", kendall_correlation)
	print("............................................................")
	print("Coeficient of determination(r_squared)")
	print("Result: ", r_value**2)
	
	# plt.xscale('log')
	# plt.yscale('log')
	#plt.scatter(list_dataX, list_dataY, )
	ax = sns.regplot(x=list_dataX, y=list_dataY, line_kws={"color":"r","alpha":0.7,"lw":2})
	ax.set(xlabel='Workload', ylabel='Failures')
	plt.savefig("corrlation.pdf")
	return 
	
	
if len(sys.argv) == 7:
	fileName1 = sys.argv[1]
	column_name1 = sys.argv[2]
	separator1 = sys.argv[3]
	fileName2 = sys.argv[4]
	column_name2 = sys.argv[5]
	separator2 = sys.argv[6]
	pearson(fileName1, column_name1, separator1,fileName2, column_name2, separator2)
else:
	print("ERROR, usage: %s <input file> <column> <separator> <imput file> <column> <separator> " % sys.argv[0])
	print("<input file>: file 1")
	print("<column>: column for correlation in file 1")
	print("<separator>: separator of colunnms.  For space use: sp")
	print("<input file>: file 2")
	print("<column>: column	 for correlation in file 2")
	print("<separator>: separator of columns.  For space use: sp")
	sys.exit(0)