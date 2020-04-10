# -*- coding: utf-8 -*-
"""
@author: heman
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar

df = pd.read_csv('hour.csv')
date = '2012-09-01'
df = df[df['dteday'] < date]
df = df.rename(columns = {'weathersit':'weather'})
num_col = ['temp','atemp','hum','windspeed','casual','registered','cnt']
cat_col = ['weather','holiday','workingday']

def set_fig(title):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.6, wspace=0.6)
    fig.set_figheight(12)
    fig.set_figwidth(12)
    fig.suptitle(title, fontsize = 20)
    
    return(fig)
    
#####################################################PART1###############################################
#Univariate Analysis
#Visualizing y
fig = set_fig('Count Distribution')
sns.distplot(df['cnt'])
plt.savefig('Plots/Count Distribution')

#All other numerical distributions
fig = set_fig('Other Distributions')
ncol = num_col[:-1]
i = 1
for col in ncol:
    plt.subplot(3,2,i)
    sns.distplot(df[col])
    i+=1
plt.savefig('Plots/Other Distributions')

#Box plot to further visualize numerical variables
fig = set_fig('Box Plots')
i = 1
for col in num_col:
    plt.subplot(3,3,i)
    sns.boxplot(y = col, data = df)
    i+=1
plt.savefig('Plots/Box Plots')

#Bar plot to visualize some categorical variables
fig = set_fig('Bar Plots')

i = 1
for col in cat_col:
    plt.subplot(2,2,i)
    
    sns.barplot(x = col, y = col, estimator = len, data = df,palette='BrBG')
    i+=1

plt.savefig('Plots/Bar Plots')

#####################################################PART2###############################################
#Analysis of Target Variable with Other Variables (Mostly Bivariate)

#Variation in Cnt by month
monthly = df.groupby(['yr', 'mnth']).mean()
x_label = []
for month in range(1, 13):
  x_label.append(calendar.month_name[month] + ' 2011')
for month in range(1, 9):
  x_label.append(calendar.month_name[month] + ' 2012')
monthly['x'] = x_label
fig = set_fig('Variation by Month')
ax1 = sns.barplot(x = 'x',y = 'cnt',data = monthly, palette='GnBu')
ax1.set_xlabel('Month')
plt.xticks(rotation = 90)

plt.savefig('Plots/Monthly Variation')

#Variation in cnt  Daily
daily = df.groupby('dteday').sum()
fig = set_fig('Daily Count Trend')
ax2 = sns.lineplot(x = daily.index,y = daily['cnt'])

plt.savefig('Plots/Daily Variation')

#Variation by Day of the Week
daily['date'] = pd.to_datetime(daily.index)
daily['weekday'] = daily['date'].dt.weekday
daily.loc[daily.holiday != 0,'holiday'] = 1
by_weekday = daily.groupby('weekday').mean()
fig = set_fig('Variation by Day of the Week')
ax3 = sns.barplot(x = by_weekday.index,y = "cnt", data = by_weekday, palette='PuBu')

plt.savefig('Plots/Variation by Day of Week')

#Variation in cnt by Hour
hourly = df.groupby('hr').mean()
hourly['hour'] = hourly.index
melted = hourly.melt(id_vars=['hour'], value_vars = ['registered', 'casual'] ) 

fig = set_fig('Hourly Variation')
ax4 = plt.subplot(2,1,1)
pal = ["#9b59b6", "#3498db"]
sns.barplot(x = 'hour',y = 'value', data = melted, hue = 'variable', palette=pal )
ax4.set_ylabel('cnt')
ax42 = plt.subplot(2,1,2)

pal = [ "#3498db","#9b59b6"]
sns.barplot(x = 'hr', y = 'cnt', data = df, hue = 'workingday', palette = pal)
ax42.set_xlabel('hour')
plt.savefig('Plots/Hourly Variation for Registered and Casual Users')

#Variation by temperature and humidity
fig = set_fig('Effect of Temperature and Humidity')
cmap = plt.get_cmap('YlOrRd')
i = 1
for attr in ['hum','temp']:
    for col in ['holiday', 'workingday']:
        x = df[df[col] == 1][['hr']]
        y = df[df[col] == 1]['cnt']
        z = df[df[col] == 1][attr]
        
        ax = plt.subplot(2,2,i)
        ax.set(xlabel = 'hour', ylabel = 'cnt')
        
        i+=1
        points = plt.scatter(x, y, c=z, s=50, cmap=cmap)
        ax.legend([points],[col], facecolor = 'white')
        fig.colorbar(points).set_label(attr)
plt.savefig('Plots/Effect of Temperature and Humidity')

fig = set_fig('Variation by Weather')
ax5 = sns.lineplot(x = 'mnth', y = 'cnt', data = df, estimator = np.average, hue = 'weather')
plt.savefig('Plots/Variation by Weather')

#####################################################PART3###############################################
#Correlation Analysis
corr_df = df[num_col + cat_col]
corr = corr_df.corr()
corr = round(corr,2)

fig = set_fig('Correlation Heatmap')

# Plot the heatmap
ax6 = sns.heatmap(corr, annot=True,
        xticklabels=corr.columns,
        yticklabels=corr.columns, cmap = 'coolwarm')
plt.savefig('Plots/Correlation Heatmap')

