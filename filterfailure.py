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
	def __init__(self, failureType, epoch):
		self.failureType = failureType
		self.startTime = epoch
		#self.ftime = epoch
		self.endTime = epoch
		self.nodes = []
		self.toPrint = True
	def updateTime(self, epoch):
		if self.startTime > epoch:
			self.startTime = epoch
		if self.endTime < epoch:
			self.endTime = epoch

def failureFilter(fileName,hardware, software):
	jobs = {}
	formatAlt = '%Y-%m-%d %H:%M:%S'
  
	#size of failure file
	with open(fileName) as f:
		lines = len(f.readlines())
		
	count = 0 
	
	with open(fileName) as f:
		next(f)
		for line in f:
			count = count + 1
			print ("Progress: %d%%"% (count/lines*100),end="\r") 
	
			fields = line.split('|')
			jobid = int(fields[2].strip())
	  
			ftime = fields[3].strip() #before ftime
			failureTime = time.strptime(ftime, formatAlt)
			epoch = time.mktime(failureTime)
			category = fields[4].strip()
			reason = fields[5].strip()
			description = fields[6].strip()
			text = fields[7].strip()
	  
			# ignore heartbeat faults
			if "Heartbeat Fault" in description:
				continue
			
			# parse error number
			if(description == "GPU XID"):
				#get the number
				match = re.search(r"GPU Xid (\d+)", text)
				description = match.group(0)
	  
			node = ''
			#find the node related with the failure
			#for HT Lockup, no node information is reported
			match = re.search(r"c(\d+)-(\d+)c(\d+)s(\d+)n(\d+)", text)
			if not match:
				match = re.search(r"c(\d+)-(\d+)c(\d+)s(\d+)", text)
				if match:
					node = match.group(0)
			else:	
				node =  match.group(0)
	  
			entry = FailedJob(description, epoch)
			#create entry
			if jobid not in jobs:
				jobs[jobid] = {}
			if ftime not in jobs[jobid]:
				jobs[jobid][ftime] = {}
			if category not in jobs[jobid]:
				jobs[jobid][ftime][category] = {}
			if description not in jobs[jobid][ftime][category]:
				jobs[jobid][ftime][category][description] = FailedJob(description, epoch)
			else:
				jobs[jobid][ftime][category][description].updateTime(epoch)

			if node != '':
				if node not in jobs[jobid][ftime][category][description].nodes:
					jobs[jobid][ftime][category][description].nodes.append(node);
	  
			#ignore reason related errors
			#if description == "Out of Memory":
			#jobs[jobid]['software'][description].toPrint = False
			#continue
			#if description == "GPU Xid 31":
			#jobs[jobid]['software'][description].toPrint = False
			if category == 'hardware':
				jobs[jobid][ftime][category][description].toPrint = hardware
			else:	 
				jobs[jobid][ftime][category][description].toPrint = software
	return jobs

def printEntry(jobid, ftime, category, description, nodes, startTime, endTime):
  out = []
  out.append("| ")
  out.append("titan")
  out.append("| ")
  out.append(str(jobid))
  out.append(" | ")
  out.append(str(ftime))
  out.append(" | ")
  out.append(category)
  out.append(" | ")
  out.append(description)
  out.append(" | ")
  out.append(str(startTime))
  out.append(" | ")
  out.append(str(endTime))
  out.append(" | ")
  out.append(" ".join(jobs[jobid][ftime][category][description].nodes))
  print ("\t".join(out))
  file.write("\n")
  file.write(" ".join(out))
 
def checkRedundant(key1, key2, table):
  if key1 in table and key2 in table:
    if table[key1].toPrint and table[key2].toPrint:
        node1 = table[key1].nodes[0]
        node2 = table[key2].nodes[0]
        if node1 == node2:
	        return True
  return False
     	

def outputAll(jobs):
	for jobid in jobs:
		for ftime in jobs[jobid]:	
			for category in jobs[jobid][ftime]:
				for description in jobs[jobid][ftime][category]:
					if jobs[jobid][ftime][category][description].toPrint:
						#if description == "GPU Xid 31" or description == "Out of Memory":
						#if description == "Out of Memory":
						printEntry(jobid, ftime, category, description, jobs[jobid][ftime][category][description].nodes, jobs[jobid][ftime][category][description].startTime, jobs[jobid][ftime][category][description].endTime) 
				

def output(jobs):
  for jobid in jobs:
    #cross check, if the same node reports both software and hardware
    #failures, remove the software failure entry
    if len(jobs[jobid]) == 2:
        hardfailure_nodes = []
        #both hardware and software failures reported
        for description in jobs[jobid]['hardware']:
	        hardfailure_nodes.extend(jobs[jobid]['hardware'][description].nodes)
      
        softwareFailures = jobs[jobid]['software']
        for description in softwareFailures:
	        remove = True
	        nl = list(jobs[jobid]['software'][description].nodes)
	        for node in nl:
	            if node not in hardfailure_nodes:
	                remove = False
	            else:
	                if node in jobs[jobid]['software'][description].nodes:
	                    jobs[jobid]['software'][description].nodes.remove(node)

	        if remove:
	            jobs[jobid]['software'][description].toPrint = False
	  
    #check redundant software failures
    if 'software' in jobs[jobid] and len(jobs[jobid]['software']) > 1:
        key1 = 'GPU BUS'
        key2 = 'GPU Xid 62'
        if checkRedundant(key1, key2, jobs[jobid]['software']):
	        jobs[jobid]['software'][key2].toPrint = False
    
    #check redundant hardware failures
    if 'hardware' in jobs[jobid] and len(jobs[jobid]['hardware']) > 1:
        key1 = 'GPU DBE'
        key2 = 'GPU DPR'
        if checkRedundant(key1, key2, jobs[jobid]['hardware']):
	        jobs[jobid]['hardware'][key2].toPrint = False
    
	
    outputAll(jobs)
	
if len(sys.argv) > 4:
	fileName = sys.argv[1]
	outputFileName =  sys.argv[2]
	hardware =  sys.argv[3]
	software =  sys.argv[4]
	file = open(outputFileName, 'w')
	
	file.write("|host name  |Job ID  |   FailTime            |   Category  |  Description   |   StartTime   |   EndTime   |    Nodes")
	jobs = failureFilter(fileName, hardware, software)
	outputAll(jobs)
	file.close()
else:	
	print ("ERROR, usage: %s <file><output file><hardare><software>" % sys.argv[0])
	print ("<file>: Failure log file")
	print ("<output file>: file name to output filtered information")
	print ("<hardware>: True or False")
	print ("<Software>: True or False")
	sys.exit(0)
