# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 21:13:47 2017

@author: Max
"""

import unicodedata
from unidecode import unidecode
from urllib.request import  urlopen
import bs4 as bs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import style


style.use('fivethirtyeight')

# Scrape data from the web

sauce = urlopen('http://www.tatort-fundus.de/web/rangliste/folgen-wertungen/rangliste-auswertung/aktueller-zeitpunkt.html')
soup = bs.BeautifulSoup(sauce,'lxml')

table = soup.find('table')

table_rows = table.find_all('tr')
j = 0 
table1 = []

# Parse data into rows

for tr in table_rows:
    td = tr.find_all('td')
    row = [i.text for i in td]
    table1.append(row)
    j+=1
    
# Use only Rows 8 through 1034 since only these contain data

data = table1[8:1034]
headers = table1[7]
headers[5] = 'Bewertungen'
headers[8] = 'Kommentare'

# Create dataframe from raw data
       
df = pd.DataFrame(data = data, columns = headers)

# Convert variables that are numeric to numeric

a = ['lfd Nr.','Nr.','Schnitt','Bewertungen','Mio','Quote']
b = [3,2]


for i in a:
    df[i] = pd.to_numeric(df[i], errors = 'coerce')
    
print (10*'=' + 'Conversion of numeric values complete' + 10*'=')

# Convert string values in dataframe to string

for j in b:
    for row_num in range(len(df.iloc[:,j])):
        ex = unidecode(df.iloc[row_num,j])
        df.iloc[row_num,j] = ex
    
    
print (10*'=' + 'Conversion of string values complete' + 10*'=')

print (df.head(7))

df = df[np.isfinite(df['lfd Nr.'])]

# Create Column Staffel and set equal to 0

df['Staffel'] = 0
    
# Create dictionary of episode number related to year aired
   
season_dict = [[969, 929, 893, 857, 822, 786, 751, 717, 686, 651, 619, 584,
              554, 521, 490, 461, 432, 403, 377, 350, 323, 301, 286, 268, 253,
              238, 227, 215, 201, 189, 177, 165, 153, 144, 132, 120, 108, 95,
              83, 70, 59, 47, 36, 26, 14, 3, 1], list(reversed(range(1970,2017)))]

season = pd.DataFrame(season_dict)
season = season.T

for i in range(len(df)):
    for j in range(len(season)):
        if df.iloc[i,1] >= season.iloc[j,0]:
            df.iloc[i,9] = season.iloc[j,1]
            break

df.head()

### Analyse pro Jahr

dfy = df.groupby(['Staffel'], as_index = False)['Quote','Schnitt'].mean()

### Durchschnitt pro Jahr

dfy.head()

### Plot of average viewer quota over the years

fig, ax = plt.subplots()

ax.plot(dfy.Staffel, dfy.Quote, color = '#2835a3')
ax.spines['right'].set_visible(True)
ax.spines['top'].set_visible(True)

ax.set_xlim(1969,2018)
ax.set_ylim(0,100)
ax.set_xlabel('Year')
ax.set_ylabel('Quota of viewers in %')

plt.show()

### Export to csv

dfy.to_csv("C:/Users/Max/Documents/Blog/Tatort/time.csv", index = False, columns = ['Staffel','Quote'])


### Fit a normal distribution to histogram

from scipy.stats import norm

mu, std = norm.fit(df.Schnitt)


### Plot distribution of ratings

fig, ax = plt.subplots()

n, bins, patches = plt.hist(df.Schnitt, 15, normed=1, facecolor='blue', alpha=0.75)

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)    
p = norm.pdf(x, mu, std)
plt.plot(x,p,'darkblue', linewidth = 2)


ax.set_xlabel('Average user rating')
plt.show()

### Save data as csv 

np_hist = np.histogram(df.Schnitt, bins = 15, normed = True)
df_hist = pd.DataFrame([np_hist[1],np_hist[0]]).T.fillna(0)
df_hist.columns = ['x','Schnitt']

df_hist.to_csv("C:/Users/Max/Documents/Blog/Tatort/hist.csv", index = False, columns = ['x','Schnitt'])

d = {'x': x, 'p' : p}
df_norm = pd.DataFrame(data = d)
df_norm.to_csv("C:/Users/Max/Documents/Blog/Tatort/hist_norm.csv", index = False)



### Use only part of the data from here on 

df = df[(df['Staffel'] >= 2000)]

### Plot distribution of and quotas & fit a normal distribution to histogram

mu, std = norm.fit(df.Quote[np.isnan(df.Quote)==False])
fig, ax = plt.subplots()

n, bins, patches = plt.hist(df.Quote[np.isnan(df.Quote) == False],25, normed=1, facecolor='purple', alpha=0.75)

xmin, xmax = plt.xlim()
x2 = np.linspace(xmin, xmax, 100)    
p2 = norm.pdf(x2, mu, std)
plt.plot(x2,p2,'darkblue', linewidth = 2)

ax.set_xlabel('Average viewer quota (in %)')
plt.show()



### Plot rating vs. viewer quota

Erm_set = set(list(df.Ermittler))

Erm_set_sorted = sorted(Erm_set)
Erm_list = [i  for i in range(len(Erm_set))]
Erm_color = [j for j in Erm_list]

#print Erm_set_sorted
print (len(Erm_set_sorted))
#print Erm_color

color_dict = {}
for i in range(len(Erm_list)):
    color_dict[Erm_set_sorted[i]] = Erm_color[i]
    
### Create dictionary to associate each 'Ermittler' with a unique color


df['color'] = df['Ermittler']
df['color'].replace(color_dict, inplace = True)


### For this part use only episodes post 2000

df = df[df.Staffel >= 2000]

df.head()

X = df['Quote']
Y = df['Schnitt']
color = df['color']

print (df.head())

print (X.dtypes, len(X))
print (Y.dtypes, len(Y))
print (color.dtypes, len(color))


# Make plot with horizontal colorbar
fig, ax = plt.subplots()

cax = plt.scatter(X,Y, c = color, s = 30)

cbar = fig.colorbar(cax, ticks=Erm_color, orientation='vertical')
cbar.ax.set_yticklabels(Erm_set_sorted)


plt.xlabel('Quota of viewers in %')
plt.ylabel('Average user rating')
plt.axis([15,40,3,9])
plt.show()


outliers = df.sort_values(by = 'Quote', ascending = False)
outliers.head(10)


corr_mat = df[['Schnitt','Quote']].corr()


### Export to csv

df.to_csv("C:/Users/Max/Documents/Blog/Tatort/xy.csv", index = False, columns = ['Quote','Schnitt','Ermittler','color'])