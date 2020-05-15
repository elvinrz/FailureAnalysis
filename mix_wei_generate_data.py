from reliability.Fitters import Fit_Weibull_Mixture, Fit_Weibull_2P
from reliability.Distributions import Weibull_Distribution
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.stats import weibull_min
import sys


def generate_data(output_dir, CPU_factor, GPU_factor, MEM_factor):

	'''
	#generated wit mixture weibull in R
	alpha_mem = 1208
	beta_mem  = 0.871

	alpha_gpu = 59666
	beta_gpu  = 0.934

	alpha_cpu = 289484
	beta_cpu  = 2.778
	'''

	##alpha and beta of real data
	#'''
	alpha_mem = 295647
	beta_mem = 0.918
	
	alpha_gpu = 44721
	beta_gpu = 0.551

	alpha_cpu = 1595690
	beta_cpu = 0.758
	

	#'''
	####################################################################
	#extract real mtbf data
	x_cpu = []
	x_gpu = []
	x_mem = []
	file_name = "../Data/Titan_Data/GPU_mtbf_epoch1.txt"
	with open(file_name) as log:
		for line in log:
				x_gpu.append(int(line))
	
	file_name = "../Data/Titan_Data/CPU_mtbf_epoch1.txt"
	with open(file_name) as log:
		for line in log:
				x_cpu.append(int(line))

	file_name = "../Data/Titan_Data/MEM_mtbf_epoch1.txt"
	with open(file_name) as log:
		for line in log:
			x_mem.append(int(line))

	n_samples_gpu = int(int(len(x_gpu)) * float(GPU_factor))
	n_samples_cpu = int(int(len(x_cpu)) * float(CPU_factor))
	n_samples_mem = int(int(len(x_mem)) * float(MEM_factor))
	
	

	gpu = Weibull_Distribution(alpha=alpha_gpu,beta=beta_gpu).random_samples(n_samples_gpu)
	cpu = Weibull_Distribution(alpha=alpha_cpu,beta=beta_cpu).random_samples(n_samples_cpu)
	mem = Weibull_Distribution(alpha=alpha_mem,beta=beta_mem).random_samples(n_samples_mem)

	shape, loc, scale = ss.weibull_min.fit(gpu, floc=0)
	mean_gpu, var = weibull_min.stats(shape,loc,scale, moments='mv')
	print("gpu total failures = "+ str(n_samples_gpu))
	print("gpu shape = " + str(shape))
	print("gpu scale = " + str(scale))
	print("gpu Mean: "+str(mean_gpu))
	print("-----------------")
	shape, loc, scale = ss.weibull_min.fit(cpu, floc=0)
	mean_cpu, var = weibull_min.stats(shape,loc,scale, moments='mv')
	print("cpu total failures = "+ str(n_samples_cpu))
	print("cpu shape = " + str(shape))
	print("cpu scale = " + str(scale))
	print("cpu Mean: "+str(mean_cpu))
	print("-----------------")
	shape, loc, scale = ss.weibull_min.fit(mem, floc=0)
	mean_mem, var = weibull_min.stats(shape,loc,scale, moments='mv')
	print("mem total failures = "+ str(n_samples_mem))
	print("mem shape = " + str(shape))
	print("mem scale = " + str(scale))
	print("mem Mean: "+str(mean_mem))

	#new proportions
	t = 0
	t = n_samples_mem+n_samples_cpu+n_samples_gpu
	
	proportion_cpu = n_samples_cpu / t
	proportion_gpu = n_samples_gpu / t
	proportion_mem = n_samples_mem / t

	mixture_mean = mean_gpu * proportion_gpu + mean_mem * proportion_mem + mean_cpu * proportion_cpu
	print("-----------------")
	print("Total failures = "+ str(n_samples_mem+n_samples_cpu+n_samples_gpu))
	print("mixture mean without mixture= " + str(mixture_mean/60/60))

	print("-----------------")
	xvals = np.linspace(0,50,t)
	wei_cpu = Weibull_Distribution(alpha=alpha_cpu,beta=beta_cpu).CDF(xvals=xvals,show_plot=False)
	wei_gpu = Weibull_Distribution(alpha=alpha_gpu,beta=beta_gpu).CDF(xvals=xvals,show_plot=False)
	wei_mem = Weibull_Distribution(alpha=alpha_mem,beta=beta_mem).CDF(xvals=xvals,show_plot=False)
	Mixture_CDF = wei_gpu * proportion_gpu + wei_cpu * proportion_cpu + wei_mem * proportion_mem
	shape, loc, scale = ss.weibull_min.fit(Mixture_CDF, floc=0)
	meanw, var = weibull_min.stats(shape,loc,scale, moments='mv')
	print("Mix Mean: "+str(meanw/60/60))


	#save all data (cpu + gpu + mem)
	all_data = np.hstack([gpu,cpu,mem])
	file = open(output_dir + "TOTAL_mtbf_epoch1_exascale.txt", 'w+')
	for i in all_data:
		if int(i) != 0:
			file.write(str(int(i))+"\n")

	file.close()
	sys.exit()



if len(sys.argv) >= 5:
	output_dir = sys.argv[1]
	GPU_factor = sys.argv[2]
	CPU_factor = sys.argv[3]
	MEM_factor = sys.argv[4]
	generate_data(output_dir, CPU_factor, GPU_factor, MEM_factor)
	
else:
	print ("ERROR")
	sys.exit(0)
