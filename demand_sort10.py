# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import time
import datetime

os.chdir('E:/Lectures/CSCE 689 697/paper/data')

## Importing the dataset in case need to transfer from xlsw format
#data_xls = pd.read_excel('Data181112.xlsx', index_col=0) # the first column is index
#data_xls.to_csv('Data181112.csv', encoding='utf-8')

dataset = pd.read_csv('201810-bluebikes-tripdata.csv')

# convert to time format
dataset['starttime']=pd.to_datetime(dataset['starttime'])


# get the station series
stationnum = list(set(dataset["start station id"]))

## sort the demand for station and then select time in year month day hour min sec 
start = time.time()
month_num=10
demand =np.zeros((len(stationnum),720))
station_index=len(stationnum)-1
for station in range(0,station_index):   
    demand_per_s=dataset[dataset["start station id"] == stationnum[station]] # sort the demand at a station
    for day in range(0,29):# the acutal day is day +1
        for hour in range(0,23):    
            demand_per_hmin=demand_per_s[demand_per_s['starttime']>=pd.datetime(2018,month_num,day+1,hour,0,0)] # sort demand during an hour at a station
            demand_per_h=demand_per_hmin[demand_per_hmin['starttime']<=pd.datetime(2018,month_num,day+1,hour,59,59)]
            demand[station,day*24+hour]=len(demand_per_h)


end = time.time()
print(str(end-start))

df = pd.DataFrame(demand)
df.to_csv('demand10.csv')
