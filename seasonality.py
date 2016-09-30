import numpy as np
import pandas as pd
import math
from scipy.stats import norm
from matplotlib import pyplot as plt 
from scipy import signal as sn
from scipy.stats import norm


filepath = "user_1005_3108.txt"

def linear_smooth(arr):
    # Compute trend using linear smooth
    n = len(arr)
    coeff = np.polyfit(range(0,n), arr, 1)
    
    trend_line = []
    for i in range(0,n):
        trend_line.append(i * coeff[0] + coeff[1])
        
    remain_comp = lambda i: arr[i]/trend_line[i], range(0,n)
    return (trend_line, remain_comp)


def cal_acf(data):
    ' Calculate the autocorrelation function of data
    ' data: input series
    ' output: acf coefficients 
    n = len(data)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def acf(h):
        acf_lag = ((data[:n - h] - np.mean(data[:n - h])) * (data[h:] - np.mean(data[h:]))).sum() / float(n) / c0
        return round(acf_lag, 3)
    
    x = np.arange(12) # Avoiding lag 0 calculation
    acf_coeffs = map(lambda i: acf(i), x)
    return acf_coeffs

def complex_smooth(arr):
    ' Compute trend using savgol_filter
    trend = sn.savgol_filter(arr,7,1)
    remain_comp = lambda i: arr[i]/trend[i], range(0,n)
    return (trend, remain_comp)

def my_ma(arr):
    ' Compute trend using moving average
    
    trend = np.empty(len(arr))
    trend.fill(np.NaN)
    remain_comp = trend
    
    for i in range(6, len(arr_ma)):
        trend[i] = np.mean(arr[i-6:i+1])
    
    remain_comp = lambda i: arr[i]/trend[i], range(0,n)
    return (trend,remain_comp)

def determine_season(arr, conf_interval):
    ' Determine if time series arr has seasonal component
    ' arr: numpy array
    ' conf_interval: confident interval
    ' Output: seasonal level
    ' Step 1: trend = smooth the series by different methods (ma, linear, complex method)
    ' Step 2: remainder = arr/trend
    ' Step 3: compute acf of the remainder
    ' Step 4: find acf value lying outside of confident interval
    
    n = len(arr)
    complex_trend = complex_smooth(arr)
    linear_trend = linear_smooth(arr)
    ma_trend = my_ma(arr)
    
    ' Step 2: remainder = arr/trend
    remain_comp = lambda i: arr[i]/complex_trend[i], range(6,n)
    remain_ma =  lambda i: arr[i]/ma_trend[i], range(6,n)
    
    ' Step 3: compute acf of the remainder
    acf = cal_acf(remain_ma)
    
    ' Step 4: find acf value lying outside of confident interval
    can_sea = [7, 30]
    sea = []
    acf_mean = np.mean(acf)
    acf_sig = 1/math.sqrt(n)
    
    cri_val_upper = acf_mean + norm.ppf(conf_interval) * acf_sig
    cri_val_lower = acf_mean - norm.ppf(conf_interval) * acf_sig
    
    for i in can_sea:
        if(acf[i] > cri_val_upper or acf[i] < cri_val_lower):
            sea.append(i)
    return sea

def main(filepath): 
    ' Determine seasonal level of a time series
    ' filepath: data file
    ' Step 1: Read data
    ' Step 2: determine seasonal level
     
    df = pd.read_csv(filepath, header=0, index_col=None)
    arr_val = df['msg']
    conf_interval = 0.95
    sea = determine_season(arr_val, conf_interval)


def plot_trend_season(trend, sea):
    ' Plot trend and seasonal component
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(trend)
    axarr[0].set_title('Trend')
    axarr[1].plot(sea)
    axarr[1].set_title('Season')

    plt.show()
    print sea

def plot_trend(arr):    
    ' Plot trends computed by different methods: moving average, linear, savgol_filter
    
    ma = my_ma(arr)
    nonlinear = complex_smooth(arr)
    linear = linear_smooth(arr)
    
    hand1, = plt.plot(list(arr), label='series')
    hand2, = plt.plot(nonlinear, label='savgol_filter')
    hand3, = plt.plot(linear, label='linear')
    hand4, = plt.plot(ma, label='ma')
    plt.legend([hand1, hand2, hand3, hand4], ['series', 'savgol_filter', 'linear', 'ma'])
    plt.show()
    
main(filepath)
    
    