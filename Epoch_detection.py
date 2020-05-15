import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates



#Colecting data

market_data = pd.read_csv("gpu.csv",parse_dates = ['Week'], index_col=[0], header = 0,dayfirst=True)

pd.set_option("display.max_rows", None)

print(market_data)

#print(data['Value'])
#print(market_data['Value'])


#Calculating EMA and difference
market_data['ema'] = market_data['Value'].ewm(30).mean()
market_data['diff_pc'] = (market_data['Value'] / market_data['ema']) - 1


#Defining bull/bear signal
TH = 0
market_data['Signal'] = np.where(market_data['diff_pc'] > TH, 1, 0)
market_data['Signal'] = np.where(market_data['diff_pc'] < -TH, -1, market_data['Signal'])


# Plot data and fits

import seaborn as sns  # This is just to get nicer plots

signal = market_data['Signal']
print(market_data['Signal'])

# How many consecutive signals are needed to change trend
min_signal = 12

# Find segments bounds

bounds = (np.diff(signal) != 0) & (signal[1:] != 0)

#print(signal)

bounds = np.concatenate(([signal[0] != 0], bounds))
bounds_idx = np.where(bounds)[0]
# Keep only significant bounds
relevant_bounds_idx = np.array([idx for idx in bounds_idx if np.all(signal[idx] == signal[idx:idx + min_signal])])
# Make sure start and end are included


if relevant_bounds_idx[0] != 0:
    relevant_bounds_idx = np.concatenate(([0], relevant_bounds_idx))
if relevant_bounds_idx[-1] != len(signal) - 1:
    relevant_bounds_idx = np.concatenate((relevant_bounds_idx, [len(signal) - 1]))
#print(relevant_bounds_idx)
# Iterate segments
plt.figure(figsize=(9,3))
    
for start_idx, end_idx in zip(relevant_bounds_idx[:-1], relevant_bounds_idx[1:]):
    # Slice segment
    segment = market_data.iloc[start_idx:end_idx + 1, :]
    print(segment.index)
    x = np.array(mdates.date2num(segment.index.to_pydatetime()))
    # Plot data
    #print(x)    
    data_color = 'red' if signal[start_idx] > 0 else 'green'
    plt.plot(segment.index, segment['Value'], color=data_color, linewidth=0.5)
    # Plot fit
    coef, intercept = np.polyfit(x, segment['Value'], 1)
    fit_val = coef * x + intercept
    fit_color = 'blue' if coef > 0 else 'orange'
    plt.plot(segment.index, fit_val, color=fit_color)
    #print("---------------------------------------")
    #print(segment.index)
    #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    #print(fit_val)
    #plt.tight_layout()
    plt.savefig("fig.pdf")		
