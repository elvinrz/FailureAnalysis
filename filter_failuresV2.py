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

class FailedJob:
	def __init__(self, line):
		self.line = line
		self.nodes = {}
	
def failureFilter(fileName, outputFileName,cat):
	jobs_fail = {}
	reason_list = []
	heartbeat_count = 0
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
			jobid = int(fields[2].strip())
			#ftime = fields[3].strip() 
			category = fields[4].strip()
			reason = fields[5].strip()
			description = fields[6].strip()
			text = fields[7].strip()
			
			#count system failures
			if "system" in reason:
				count_system_failures += 1
				
			#ignore user failures
			if "user" in reason:
				reason_list.append(line.strip())
				continue
			
			# ignore heartbeat faults
			if "Heartbeat Fault" in description:
				heartbeat_count += 1
				continue
				
			
			#ignore system failures
			# if "system" in reason:
				# # reason_list.append(line.strip())
				# continue
			
			
			if cat != "-hs" and cat != "-h" and cat != "-s":
				print("Error.  Ivalid category.  Accepted categories: -hs, -h or -s")
				sys.exit()
				
			if cat == "-h":
				if category == "software":
					continue
			
			if cat == "-s":
				if category == "hardware":
					continue
				
					
			match = re.search(r"c(\d+)-(\d+)c(\d+)s(\d+)n(\d+)", text)
			if not match:
				match = re.search(r"c(\d+)-(\d+)c(\d+)s(\d+)", text)
				if match:
					node = match.group(0)
			else:	
				node =  match.group(0)
            
			if node != '':
				if jobid in jobs_fail and category in jobs_fail[jobid] and description in jobs_fail[jobid][category] and reason in jobs_fail[jobid][category][description]:
					if node not in jobs_fail[jobid][category][description][reason].nodes.keys():
						jobs_fail[jobid][category][description][reason].nodes[node]={}
			
			#to erase '|' at the end of line
			
			# print(line[-2:])
			# sys.exit()
			if line[-2:].strip() == "|":
				line = line[:-2]
				
			entry = FailedJob(line)
			#create entry
			if jobid not in jobs_fail:
				jobs_fail[jobid] = {}
			if category not in jobs_fail[jobid]:
				jobs_fail[jobid][category] = {}
			if description not in jobs_fail[jobid][category]:
				jobs_fail[jobid][category][description] = {}
			if reason not in jobs_fail[jobid][category][description]:
				jobs_fail[jobid][category][description][reason] = entry
				jobs_fail[jobid][category][description][reason].nodes[node] = {}
	
	entry_nodes = ""
	for id in jobs_fail.keys():
		for cat in jobs_fail[id].keys():
			for des in jobs_fail[id][cat].keys():
				for rea in jobs_fail[id][cat][des].keys():
					entry_nodes = "|"+" ".join(jobs_fail[id][cat][des][rea].nodes.keys()) +"\n"
					file.write(jobs_fail[id][cat][des][rea].line.strip() + entry_nodes)
	
	
	file.close()
	print("Heartbeat fault omitted: "+str(heartbeat_count))
	print("User fauilures omitted: "+str(len(reason_list)) +" - " + str((len(reason_list) / count)*100))
	print("System failures: "+str(count_system_failures))
	print("Total failures: "+str(count))
	
if len(sys.argv) > 3:
	cat =  sys.argv[1]
	fileName = sys.argv[2]
	outputFileName =  sys.argv[3]
	failureFilter(fileName, outputFileName,cat)
	
else:	
	print ("ERROR, usage: %s <category><file><output file>" % sys.argv[0])
	print ("<category>: filter by category {-hs:hardware|-h:hardare|-s:software}")
	print ("<file>: Failure log file")
	print ("<output file>: file name to output filtered information")
	sys.exit(0)
