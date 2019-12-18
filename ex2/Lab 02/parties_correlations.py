# Show scatter plot and compute correlations for pairs of parties
# Modified from a python notebook from Harel Kein
# Call libraries
import matplotlib
from scipy.stats.stats import pearsonr
from election_functions import *
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 5.0)

# Path to datafile - change to your directory!
DATA_PATH = r'C:\Users\Liav\Desktop\Uni\Lab\ex2'


############################################
# Data analysis commands below
############################################

#Read data
df_sep_raw = pd.read_csv(os.path.join(DATA_PATH, r'votes per city 2019b.csv'), encoding = 'iso-8859-8', index_col='שם ישוב').sort_index()
df_sep = df_sep_raw.drop('סמל ועדה', axis=1) # new column added in Sep 2019
df_sep_no_meta = df_sep[df_sep.columns[5:]]


# # 1. Show correlations of parties - choose 3 parties
# party_party_city_size_scatter(df_sep, 'מחל', 'פה')
# party_party_city_size_scatter(df_sep, 'אמת', 'מרצ')
# party_party_city_size_scatter(df_sep, 'מחל', 'טב')
# party_party_city_size_scatter(df_sep, 'ודעם', 'טב')
# party_party_city_size_scatter(df_sep, 'ל', 'ג')
#


#2 heatmaps  of corolations.
h=normlazied_votes(df_sep)
h=h[h.columns[5:]]
cor_mat=(party_party_corr(h,0.0125))

print(cor_mat)
print(cor_mat.corr())
# heat_map_corr(cor_mat)


'''
###new order
new_order=["ודעם","מרצ","אמת","פה","ג","שס","מחל","ל","טב","כף"]
heat_map_corr(cor_mat[new_order])

### new order sperman
heat_map_corr_sperman(cor_mat[new_order])


#3. full potintial vs real vots
full_potenitial = full_power_of_voters(h, df_sep["בזב"])
parties_bar_vs_full(df_sep_no_meta,full_potenitial,0.0325)


#3.2 full potintial vs real vots in difrent data
# Read data
df_sep_raw_1 = pd.read_csv(os.path.join(DATA_PATH, r'votes per ballot 2019b.csv'), encoding = 'iso-8859-8', index_col='שם ישוב').sort_index()
df_sep_1 = df_sep_raw_1.drop('סמל ועדה', axis=1) # new column added in Sep 2019
df_sep_no_meta_1 = df_sep_1[df_sep_1.columns[9:]]

h_1=normlazied_votes(df_sep_1)
h_1=h_1[h_1.columns[9:]]

full_potenitial = full_power_of_voters(h_1, df_sep_1["בזב"])
parties_bar_vs_full(df_sep_no_meta,full_potenitial,0.0325)

'''