# Costa Rica High Technology Center (CeNAT)
# Advanced Computing Laboratory
# Elvis Rojas, MSc (erojas@una.cr)
# Esteban Meneses, PhD (esteban.meneses@acm.org)
# Filters failures, following these rules: 
#    Groups the failures related to the same job id together
#    Groups the failures affecting multiple nodes in the same line
#    Ignores node and blade controller heartbeat fault

import sys
import re
import datetime
import time

	
def failureFilter(fileName, outputFileName):
	line_list = []
	
	file = open(outputFileName, 'w')

	#size of failure file
	with open(fileName) as f:
		lines = len(f.readlines())
	
	count_system_failures = 0
	count = 0 
	file.write("| hostname | job_id| fail_time           | category | reason | description                      | text     | jf_id   | nodes\n")
	with open(fileName) as f:
		next(f)
		for line in f:
			count = count + 1
			print ("Progress: %d%%"% (count/lines*100),end="\r") 
	
			fields = line.split('|')
			file.write(str(line.strip()))
			if len(fields) < 4:
				file.write("\n")
	
	file.close()
	
if len(sys.argv) > 2:
	fileName = sys.argv[1]
	outputFileName =  sys.argv[2]
	failureFilter(fileName, outputFileName)
	
else:	
	print ("ERROR, usage: %s <category><file><output file>" % sys.argv[0])
	print ("<file>: Failure log file")
	print ("<output file>: file name to output filtered information")
	sys.exit(0)
