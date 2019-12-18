# Show scatter plot and compute correlations for pairs of parties
# Modified from a python notebook from Harel Kein
# Call libraries
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

# 1. Read data
df_sep_raw = pd.read_csv(os.path.join(DATA_PATH, r'votes per city 2019b.csv'), encoding = 'iso-8859-8', index_col='שם ישוב').sort_index()
df_sep = df_sep_raw.drop('סמל ועדה', axis=1) # new column added in Sep 2019
df_sep_no_meta = df_sep[df_sep.columns[5:]]


# 2. Show correlations of parties - choose 3 parties
party_party_city_size_scatter(df_sep, 'מחל', 'פה')
party_party_city_size_scatter(df_sep, 'אמת', 'מרצ')
party_party_city_size_scatter(df_sep, 'מחל', 'טב')
party_party_city_size_scatter(df_sep, 'ודעם', 'טב')
party_party_city_size_scatter(df_sep, 'ל', 'ג')


'''

# Compute (and plot?) pairwise correlations
def party_party_corr(df):
    votes_per_city = df.sum(axis=1)

    n = len(names)
    corr_mat = np.zeros([n, n])

    # Here complete a code that calculates the correlation for each pair of parties

    return corr_mat



def plot_party_corr_heat_map(corr_map, names):
    rev_names = [name[::-1] for name in list(names)]
    fig, ax = plt.subplots()
    im = ax.imshow(corr_mat, cmap=plt.get_cmap('viridis'))

    # Add parties names using set_xtick, set_xticklabels
    ax.set_title("Parties pairwise correlations")

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)  # **cbar_kw)
    cbar.ax.set_ylabel('votes correlation', rotation=-90, va="bottom")


    # Here add a loop on all cells and print the actual correlations in the heatmap - use the ax.text command
    # (look at text command in matplotlib)

    plt.show()


############################################
# Data analysis commands below
############################################



# Compute and plot orrelation matrix
thresh = 0.005
votes = parties_votes(df_sep, thresh)  # total votes for each party
n = len(votes)  # number of parties
names = votes.keys()

corr_mat = party_party_corr(df_sep)
plot_party_corr_heat_map(corr_mat, names)

'''

