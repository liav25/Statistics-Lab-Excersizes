# Compare election results to socio-economic data
#
# Call libraries
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import stats
import statsmodels.api as sm

from election_functions_new import *
# Estimate and correct for voting turnout
# Modified from a python notebook from Harel Kein
# Call libraries

pd.set_option('display.max_rows', 1300)
pd.set_option('display.max_columns', 1300)

# Path to datafile - change to your directory! (both can be the same directory)
DATA_PATH = r'C:\Users\Liav\Desktop\Uni\Lab\ex4'

#parties strings dictionary
parties_dict ={'אמת' : "עבודה גשר", 'ג' : "יהדות התורה", 'ודעם'  : "הרשימה המשותפת", 'טב'  : "ימינה", 'כף'  : "עוצמה יהודית", \
 'ל'  : "ישראל ביתנו", 'מחל'  : "הליכוד", 'מרצ'  : "המחנה הדמוקרטי", 'פה'  : "כחול לבן", 'שס'  : "שס"}
big_parties_names = [parties_dict[n][::-1] for n in parties_dict]


df_hev = pd.read_csv(os.path.join(DATA_PATH, r'HevratiCalcaliYeshuvim.csv'), encoding = 'iso-8859-8', index_col='רשות מקומית').sort_index()

analysis = 'city'  # ballot
df_sep_raw = pd.read_csv(os.path.join(DATA_PATH, r'votes per '+analysis+' 2019b.csv'),\
                         encoding = 'iso-8859-8', index_col='שם ישוב').sort_index()
df_sep = df_sep_raw.drop('סמל ועדה', axis=1)  # new column added in Sep 2019
df_sep = df_sep[df_sep.index != 'מעטפות חיצוניות']
if analysis == 'city':
    first_col = 5
else:
    first_col = 9
df_sep = df_sep[df_sep.columns[first_col:]]  # removing "metadata" columns
df_sep_raw2 = df_sep_raw[df_sep_raw.index != 'מעטפות חיצוניות']

p = df_sep.sum() / df_sep.sum().sum()
big_parties = p[p>0.005].sort_values(ascending=False).keys()

##ready
# 1. Intersect election file with demography file
# use pandas intersect command for df_hev, df_sep
shared_index = list(df_hev.index.intersection(df_sep.index))
shared_cities=pd.concat([df_hev.loc[shared_index],df_sep.loc[shared_index]],axis=1)

cities_left = list(df_hev.index.difference(df_sep.index))

print("the num of shared indexes is : {}".format(len(shared_index)))
print("the num of different indexes is : {}".format(len(cities_left)))

shared_cities_for_bar=shared_cities[shared_cities.columns[13:]]
###plot of election
parties_bar_reg_pracent(shared_cities_for_bar,10)
parties_bar_double(df_sep,shared_cities_for_bar,10,["National Rate","Shared Cities Rate"])

# 2. Compute voting frequencies for each party and each eshkol:
P_dem = np.zeros([10,10])  # P_dem[i][j] is voting frequency for party j at eshkol i
for eshkol in range(10):
    # Find all cities in current eshkol. You can use np.where (similar to R which)
    cur_cities = np.where(shared_cities["מדד חברתי-"].values.astype("float")== eshkol+1)
    # Compute parties frequencies for cities in current eshkol
    P_dem[eshkol,]  = shared_cities.iloc[cur_cities][big_parties].sum()/\
                      shared_cities.iloc[cur_cities][big_parties].sum().sum()

precent_vs_eshkol=pd.DataFrame(P_dem)
precent_vs_eshkol.columns=list(big_parties)
##precebt vs eshkol is a data frame where the columns are the parties names and the row are income pracentiles
#parties_bar_by_rows(precent_vs_eshkol)
parties_bar_multible(precent_vs_eshkol)


###3.
P_dem2 = np.zeros([10,10])
for eshkol in range(10):
    # Find all cities in current eshkol. You can use np.where (similar to R which)
    cur_cities = np.where(shared_cities["מדד חברתי-"].values.astype("float")== eshkol+1)
    # Compute parties frequencies for cities in current eshkol
    P_dem[eshkol,]  = shared_cities.iloc[cur_cities][big_parties].sum()/\
                      shared_cities.iloc[cur_cities][big_parties].sum().sum()

shared_cities = shared_cities[list(shared_cities.columns[13:])+[shared_cities.columns[10]]]
t = shared_cities.groupby(['מדד חברתי-']).sum()[parties_dict.keys()]

a = t.apply(lambda x: x/x.sum(), axis=0)
a.columns = [x[::-1] for x in a.columns]
a.index.name = (a.index.name[::-1])[1:]
a.index = a.index.astype(int)
a = a.sort_index()
a.sort_index(axis=0)
axes = a.plot.bar(rot=0, subplots=True, layout=(2,5), sharey='row', sharex='row')
plt.show()



#4.
analysis = 'ballot'
df_sep_raw_b = pd.read_csv(os.path.join(DATA_PATH, r'votes per '+analysis+' 2019b.csv'),\
                         encoding = 'iso-8859-8', index_col='שם ישוב').sort_index()
df_sep_b = df_sep_raw_b.drop('סמל ועדה', axis=1)  # new column added in Sep 2019
df_sep_b = df_sep_b[df_sep_b.index != 'מעטפות חיצוניות']
if analysis == 'city':
    first_col = 5
else:
    first_col = 9
df_sep_b = df_sep_b[df_sep_b.columns[first_col:]]  # removing "metadata" columns

###calculation
df_sep_b2 = df_sep_b.loc[shared_cities.index]
p_ballot = df_sep_b.apply(lambda x: x/x.sum(),axis=1)
p_city = shared_cities_for_bar.apply(lambda x: x/x.sum(),axis=1)
df_hev_shared = df_hev.loc[shared_cities.index]



final_df = distance_heteroginity(shared_cities,p_ballot,p_city,df_hev_shared)


#print plot
heterogini_scatter_plot(final_df)

print(final_df.corr(method='spearman'))



