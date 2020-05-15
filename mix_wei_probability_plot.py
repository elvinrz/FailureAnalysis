from reliability.Fitters import Fit_Weibull_Mixture, Fit_Weibull_2P
from reliability.Distributions import Weibull_Distribution
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.stats import weibull_min
#create some failures and right censored data
#np.random.seed(2) #this is just for repeatability for this tutorial
#group_1 = Weibull_Distribution(alpha=10,beta=2).random_samples(700)
#group_2 = Weibull_Distribution(alpha=30,beta=3).random_samples(300)
#all_data = np.hstack([group_1,group_2])
#failures = []
censored = []
#threshold = 9000
#for item in all_data:
#    if item>threshold:
#        censored.append(threshold)
#    else:
#        failures.append(item)
x = []
file_name = "mtbf_total_failures_gpu-cpu-mem_epoch1.txt"
with open(file_name) as log:
    for line in log:
        #if int(line) > threshold:
        #    censored.append(threshold)
        #else:
            x.append(int(line))

#fit the Weibull Mixture and Weibull_2P
#mixture = Fit_Weibull_Mixture(failures=failures,right_censored=censored)
fig, ax = plt.subplots(figsize=(5, 2))

single = Fit_Weibull_2P(failures=x,right_censored=censored)#, show_probability_plot = False)


#plot the histogram of all the data and shade the censored part white
#n_bins = int(len(x)/100)
#N,bins,patches = plt.hist(x, density=True, alpha=0.2, color='k', bins=n_bins, edgecolor='k')
#for i in range(np.argmin(abs(np.array(bins)-threshold)),len(patches)): #this is to shade the censored part of the histogram as white
#    patches[i].set_facecolor('white')

#extract the y_vals from the mixture and build the Mixture PDF using the proportions
x = np.sort(x)
xvals2 = np.linspace(0,max(x),5211)
xvals = np.arange(1, len(x)+1)/len(x)
#xvals2 = np.arange(1, len(x)+1)


alpha_1 = 1197
beta_1  = 0.873
proportion_1 = 0.2065

alpha_2 = 60448
beta_2  = 0.928
proportion_2 = 0.7616

alpha_3 = 300016
beta_3  = 2.987
proportion_3 = 0.0319#0.02


loc = 0
wei = ss.weibull_min(beta_1, loc, alpha_1) # shape, loc, scale - creates weibull object
part_1 = wei.cdf(x)

wei = ss.weibull_min(beta_2, loc, alpha_2) # shape, loc, scale - creates weibull object
part_2 = wei.cdf(x)

wei = ss.weibull_min(beta_3, loc, alpha_3) # shape, loc, scale - creates weibull object
part_3 = wei.cdf(x)

'''
part_1 = Weibull_Distribution(alpha=alpha_1,beta=beta_1).CDF(xvals=x,show_plot=False)
part_2 = Weibull_Distribution(alpha=alpha_2,beta=beta_2).CDF(xvals=x,show_plot=False)
part_3 = Weibull_Distribution(alpha=alpha_3,beta=beta_3).CDF(xvals=x,show_plot=False)
'''
Mixture_CDF = part_1*proportion_1 + part_2*proportion_2 + part_3*proportion_3



#print("---------------------------------------------")
#print(part_1)
#print(part_2)
#print("proportion_1")
#print(mixture.proportion_1)
#print("proportion_2")
#print(mixture.proportion_2)



#plot original data
plt.plot(x, xvals, 'k-', linewidth=0.5, label="Data")


D, P = ss.kstest(x, lambda y : Mixture_CDF)
print("Mixture Weibull KS D Value: " + str(round(D,2)) + " - P value: " + str(P) )

shape, loc, scale = ss.weibull_min.fit(Mixture_CDF, floc=0)
print("shape"+str(scale))
#plot the Mixture and Weibull_2P
plt.plot(x, Mixture_CDF,linewidth=2.5, label='Mixture Weibull', color = "orange")
#Weibull_Distribution(alpha=single.alpha,beta=single.beta).CDF(xvals=xvals,label='Weibull_2P')

#---------------------------------------------------------------------------------------

D, P = ss.kstest(x, lambda y : wei.cdf(x))
print("Weibull KS D Value: " + str(round(D,2)) + " - P value: " + str(P) )


shape, loc, scale = ss.weibull_min.fit(x, floc=0)
wei = ss.weibull_min(shape, loc, scale) # shape, loc, scale - creates weibull object
#plt.plot(x, wei.cdf(x),'b--',linewidth=0.6, label="Weibull 2P")
#xvals = np.linspace(0,m,5211)




plt.ylabel("Cumulative Probability")

plt.xlabel("TBF(Seconds)")

plt.legend(edgecolor="black",prop={'size': 10})
plt.xscale('log')
#plt.show()
#plt.tight_layout()
plt.savefig("Probability_plot.pdf")

#print the goodness of fit measure
#print('Weibull_2P BIC:',single.BIC)
#print('Weibull_2P AIC:',single.AICc)
