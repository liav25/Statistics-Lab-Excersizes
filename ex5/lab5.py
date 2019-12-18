# Compare election results to socio-economic data
#
# Call libraries
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import stats
import statsmodels.api as sm
from sklearn.decomposition import PCA
from election_functions_new import *






# Estimate and correct for voting turnout
# Modified from a python notebook from Harel Kein
# Call libraries

pd.set_option('display.max_rows', 1300)
pd.set_option('display.max_columns', 1300)

# Path to datafile - change to your directory! (both can be the same directory)
DATA_PATH = r'C:\Users\Liav\Desktop\Uni\Lab\ex5'

#parties strings dictionary
parties_dict ={'אמת' : "עבודה גשר", 'ג' : "יהדות התורה", 'ודעם'  : "הרשימה המשותפת", 'טב'  : "ימינה", 'כף'  : "עוצמה יהודית", \
 'ל'  : "ישראל ביתנו", 'מחל'  : "הליכוד", 'מרצ'  : "המחנה הדמוקרטי", 'פה'  : "כחול לבן", 'שס'  : "שס"}
big_parties_names = [parties_dict[n][::-1] for n in parties_dict]



analysis = 'ballot'  # ballot
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


####leaving only big partys
p = df_sep.sum() / df_sep.sum().sum()
big_parties = p[p>0.005].sort_values(ascending=False).keys()
df_sep = df_sep[big_parties]

p_ballot = df_sep.apply(lambda x: x/x.sum(),axis=1).dropna()
names = p_ballot.columns


pca_ballot = PCA(n_components=2)  # define PCA object

###1
p_ballot_t = p_ballot.transpose()
principalComponents = pca_ballot.fit_transform(p_ballot_t)  # fit model. Compute principal components
X_pca = pca_ballot.transform(p_ballot_t)  # Perform PCA transformation
scatter_twoway(X_pca,names=names)

#####2
principalComponents = pca_ballot.fit_transform(p_ballot)  # fit model. Compute principal components
X_pca = pca_ballot.transform(p_ballot)  # Perform PCA transformation
X_pca=pd.DataFrame(X_pca)
p_ballot=pd.DataFrame(p_ballot)
X_pca.index=p_ballot.index
scatter_twoway_by_two_groups(X_pca,cities=["יהוד-מונוסון","אור יהודה"])


####3.a
analysis2="city"
df_sep_raw_cities = pd.read_csv(os.path.join(DATA_PATH, r'votes per '+analysis2+' 2019b.csv'),\
                         encoding = 'iso-8859-8', index_col='שם ישוב').sort_index()
df_sep_cities = df_sep_raw_cities.drop('סמל ועדה', axis=1)  # new column added in Sep 2019
df_sep_cities = df_sep_cities[df_sep_cities.index != 'מעטפות חיצוניות']
df_sep_cities = df_sep_cities[df_sep_cities.columns[5:]]  # removing "metadata" columns

big_parties = list(parties_votes(df_sep_cities,10).index)
df_sep_cities = df_sep_cities[big_parties]

df_sep_raw2_c = df_sep_raw_cities[df_sep_raw_cities.index != 'מעטפות חיצוניות']

df_apr_raw = pd.read_csv(os.path.join(DATA_PATH,"votes per city 2019a.csv"),
                     encoding = 'iso-8859-8', index_col='שם ישוב').sort_index()
df_apr = df_apr_raw.drop('סמל ישוב', axis=1)  # new column added in Sep 2019
df_apr = df_apr[df_apr.index != 'מעטפות חיצוניות']
df_apr = df_apr[df_apr.columns[4:]]  # removing "metadata" columns

shared = list(df_sep_cities.index.intersection(df_apr.index))
df_apr = df_apr.loc[shared].sort_index()
df_sep_cities = df_sep_cities.loc[shared].sort_index()

###leaving only big partys
p_city = df_sep_cities.apply(lambda x: x/x.sum(),axis=1)
names = p_city.columns

pca_city = PCA(n_components=2)  # define PCA object
p_city=p_city.dropna().sort_index()

pca_city.fit_transform(p_city)  # fit model. Compute principal components
X_pca = pca_city.transform(p_city) # Perform PCA transformation


q_mat1 = df_sep_raw2_c["בזב"]/df_sep_raw2_c["בזב"].sum()
scatter_twoway(X_pca,s1=q_mat1, label="Cities Voting PCA Bechirot Septmeber 19")

####3.b

#df_apr_top_14 = list(parties_votes(df_apr,14).index)

#df_apr_raw2 = df_apr_raw[df_apr_raw.index != 'מעטפות חיצוניות']

df_apr_p = df_apr.apply(lambda x: x/x.sum(),axis=1)

#3.c
df_apr_p_fix = fixing_sep_to_apr(df_apr_p, big_parties )
names2 = df_apr_p_fix.columns



# =============================================================================
 
#3.d
X_pca2 = pca_city.transform(df_apr_p_fix) # Perform PCA transformation
#
#
q_mat2 = df_apr_raw["בזב"]/df_apr_raw["בזב"].sum()
R = 10000

df_arrows = df_sep_cities.reset_index()
arr_idx = list(df_arrows[df_arrows.sum(axis=1)>10000].index)

dist_lst = [0]*(len(X_pca))
for i in range(len(X_pca)):
    dist_lst[i] = calc_dist(X_pca[i][0],X_pca[i][1],X_pca2[i][0],X_pca2[i][1])
 

# Indices of N largest elements in list 
# using sorted() + lambda + list slicing 
res = sorted(range(len(dist_lst)), key = lambda sub: dist_lst[sub])[-3:] 

#
scatter_twoway(X_pca,X_pca2,s1=q_mat1,s2=q_mat2, label="Cities Voting PCA Bechirot Septmeber vs. April 19", 
               arrows=arr_idx+res               ,longest=res)
# =============================================================================

    
   
    
    
longest_sep = p_city.iloc[res][names]
longest_apr = df_apr_p_fix.iloc[res][names]

d = {"a":1,"b":2}
d["a"]

#########
for i in range(3):
    X = list(longest_sep.columns)
    Y = longest_sep.iloc[0]
    Z = longest_apr.iloc[0]
    _X = np.arange(len(X))
    
    plt.bar(_X - 0.2, Y, 0.4, label="September 2019")
    plt.bar(_X + 0.2, Z, 0.4,label = "April 2019")
    plt.xticks(_X, X[::-1]) # set labels manually
    plt.title("Distribution of Voting in "+str(longest_sep.index[i][::-1]))
    plt.legend()
    plt.show()


