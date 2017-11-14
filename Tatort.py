

import unicodedata
from unidecode import unidecode
from urllib.request import  urlopen
import bs4 as bs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import style

import plotly.plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

plotly.offline.init_notebook_mode()

style.use('ggplot')

sauce = urlopen('http://www.tatort-fundus.de/web/rangliste/folgen-wertungen/rangliste-auswertung/aktueller-zeitpunkt.html')
soup = bs.BeautifulSoup(sauce,'lxml')

table = soup.find('table')

table_rows = table.find_all('tr')
j = 0 
table1 = []

for tr in table_rows:
    td = tr.find_all('td')
    row = [i.text for i in td]
    table1.append(row)
    j+=1
    




data = table1[8:1019]
headers = table1[7]
headers[5] = 'Bewertungen'

df = pd.DataFrame(data, columns = headers)



a = ['lfd Nr.','Nr.','Schnitt','Bewertungen','Mio','Quote']
b = [3,2]


for i in a:
    df[i] = pd.to_numeric(df[i], errors = 'coerce')
    
print (10*'=' + 'Conversion of numeric values complete' + 10*'=')

for j in b:
    for row_num in range(len(df.iloc[:,j])):
        ex = unidecode(df.iloc[row_num,j])
        df.iloc[row_num,j] = ex
    
    
print (10*'=' + 'Conversion of string values complete' + 10*'=')

print df.head(7)


df['Staffel'] = 0
    
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


# In[ ]:

### Aggregate data to seasons and compete average quota and average number of viewers

dfy = df.groupby(['Staffel'], as_index = False)['Quote','Schnitt'].mean()

### Display average quota and average number of viewers

print (df['Quote'], df['Schnitt'], len(dfy))

dfy.head()

### Plot of average viewer quota over the years



df = df[(df['Staffel'] >= 2000)]

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


df.head()

X = df['Quote']
Y = df['Schnitt']
color = df['color']

print (df.head())

print (X.dtypes, len(X))
print (Y.dtypes, len(Y))
print (color.dtypes, len(color))


# In[131]:

# Make plot with horizontal colorbar
fig, ax = plt.subplots()

cax = plt.scatter(X,Y, c = color, s = 30)
ax.set_title('Tatort Zuschauerquoten und Bewertungen')

cbar = fig.colorbar(cax, ticks=Erm_color, orientation='vertical')
cbar.ax.set_yticklabels(Erm_set_sorted)


plt.xlabel('Zuschauerquote in %')
plt.ylabel('Zuschauerbewertung')
plt.axis([0,50,0,10])
plt.show()


### Analyse pro Ermittler

dfm = df.groupby(['Ermittler'], as_index = False)['Quote','Schnitt','color'].mean()
dfm['sign'] = 0
dfm.head()

### Der Durchschnitts-Tatort

Quote_mean = dfm.Quote[:].mean()
Schnitt_mean = dfm.Schnitt[:].mean()

print (Quote_mean, Schnitt_mean, len(dfm))

dfm.loc[len(dfm)] = ['Durchschnitt', Quote_mean, Schnitt_mean, 1, 4]

dfm.tail()


# In[140]:

# Create a trace
scatter2 = go.Scatter(
    x = dfm.Quote,
    y = dfm.Schnitt,
    mode = 'markers',
    marker = dict(color = dfm.color, colorscale='Viridis', symbol = dfm.sign, size = dfm.sign*2+8, opacity = 1),
    text = dfm.Ermittler,
    
)



#Create layout
layout2= go.Layout(
    title= 'Fernsehquote und Bewertungen von Tatorten',
    hovermode= 'closest',
    xaxis= dict(
        range = [16,30],
        title= 'Zuschauerquote in %'
    ),
    yaxis=dict(
        range = [4,9],
        title= 'Zuschauerbewertung',
    ),
    showlegend= False
    )

data2 = [scatter2]


# Plot and embed in ipython notebook!
fig= go.Figure(data= data2, layout=layout2)
iplot(fig)


# In[ ]:



