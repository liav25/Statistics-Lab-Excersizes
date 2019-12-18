# Functions for Loading and analysis of election data
# Modified from a python notebook from Harel Kein
# Call libraries
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import gridspec



pd.set_option('display.max_rows',10000000)


# Path to datafile - change to your directory!
DATA_PATH = 'C:/Users/USER/PycharmProjects/Statistics_lab/ex2'


# Functions
# Get number of votes of all parties
def parties_votes_total(df, thresh):
    par = df.sum().sort_values(ascending=False)
    return par[par > thresh]


# Get votes of all parties (normalized)
def parties_votes(df, thresh):
    par = df.sum().div(df.sum().sum()).sort_values(ascending=False)
    return par[par > thresh]


# Bar plot for all parties with votes above a threshold
def parties_bar(df, thresh, city):
    width = 0.3
    votes = parties_votes(df, thresh)  # total votes for each party
    n = len(votes)  # number of parties
    names = votes.keys()

    rev_names = [name[::-1] for name in list(names)]
    fig, ax = plt.subplots()  # plt.subplots()

    city_votes = df.loc[city,names] / df.loc[city,names].sum()
    all_bar = ax.bar(np.arange(n), list(votes), width, color='b')
    city_bar = ax.bar(np.arange(n)+width, list(city_votes), width, color='r')

    ax.set_ylabel('Votes percent')
    ax.set_xlabel('Parties Names')
    ax.set_title('Votes percent per party 2019')
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(rev_names)
    ax.legend((all_bar[0], city_bar[0]), ('Israel', city[::-1]))
    plt.show()

    return fig, ax


# Plot histogram of votes for a particular party across all cities
def one_party_hist(df, party, nbins):
    votes_per_city = df.sum(axis=1)
    party_share = df[party] / votes_per_city

    plt.hist(party_share, nbins)
    plt.xlabel('Num. Votes')
    plt.ylabel('Freq.')
    plt.title('Histogram of ' + party[::-1])
    plt.show()


# Show party votes vs. city size
def party_size_scatter(df, party):
    votes_per_city = df.sum(axis=1)
    party_share = df[party] / votes_per_city

    plt.scatter(votes_per_city, party_share)
    plt.xlabel('Total Votes')
    plt.ylabel('Party %')
    plt.title('Votes for ' + party[::-1])
    plt.show()



# Show party votes for two parties
def party_party_scatter(df, party1, party2):
    votes_per_city = df.sum(axis=1)
    party_share1 = df[party1] / votes_per_city
    party_share2 = df[party2] / votes_per_city

    plt.scatter(party_share1, party_share2)  # Here draw circles with area proportional to city size
    plt.xlabel(party1[::-1])
    plt.ylabel(party2[::-1])
    plt.title('Scatter for two parties ' )
    plt.show()

#rescaling a column
def rescaling(column , a, b):
    min_value = column.min()
    max_value = column.max()
    x_norm = column.apply( lambda x: (b-a)*((x-min_value)/(max_value-min_value)) + a )
    return x_norm

def party_party_city_size_scatter(df, party1, party2,a=0,b=200):
    votes_per_city = df.sum(axis=1)
    party_share1 = df[party1] / votes_per_city
    party_share2 = df[party2] / votes_per_city
    norm_votes = rescaling(df['מצביעים'], a, b)

    alphas = rescaling(norm_votes,0.7,0.5)
    rgba_colors = np.zeros((1214, 4))
    # for red the first column needs to be one
    rgba_colors[:, 0] = 0.3
    rgba_colors[:, 2] = 0.7

    # the fourth column needs to be your alphas
    rgba_colors[:, 3] = alphas


    plt.scatter(party_share1, party_share2, s=norm_votes, color=rgba_colors)  # Here draw circles with area proportional to city size
    plt.xlabel(party1[::-1])
    plt.ylabel(party2[::-1])
    plt.title('Scatter for two parties, Dot proportional to total voters')
    plt.show()

def estimate_total_votes(df):
    n_dot = df['כשרים']
    n_tilda = df['בזב']
    n = df['בזב'].sum()



def rever(strings):
    return [x[::-1] for x in strings]



# Compute (and plot?) pairwise correlations
def party_party_corr(df,thresh):
   # votes_per_city = df.sum(axis=1)
    votes = parties_votes(df, thresh)  # parties that above thresh hold
    n = len(votes)  # number of parties
    names = votes.keys()
    new_df=df[names]
    return new_df



def heat_map_corr(cor_mat):
    f = plt.figure(figsize=(10, 10))
    plt.matshow(cor_mat.corr(), fignum=f.number, cmap='winter')
    plt.xticks(range(cor_mat.shape[1]), rever(list(cor_mat.columns)), fontsize=14)
    plt.yticks(range(cor_mat.shape[1]), rever(list(cor_mat.columns)), fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)

    plt.title('Correlation Matrix', fontsize=12)
    plt.gcf().set_facecolor("#009ACD")

    for (i, j), z in np.ndenumerate(cor_mat.corr()):
        plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')


    plt.show()



def heat_map_corr_sperman(cor_mat):
    f = plt.figure(figsize=(10, 10))
    plt.matshow(cor_mat.corr(method="spearman"), fignum=f.number, cmap='winter')
    plt.xticks(range(cor_mat.shape[1]), rever(list(cor_mat.columns)), fontsize=14)
    plt.yticks(range(cor_mat.shape[1]), rever(list(cor_mat.columns)), fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.title('Sperman correlation Matrix', fontsize=12)
    plt.gcf().set_facecolor("#009ACD")

    for (i, j), z in np.ndenumerate(cor_mat.corr()):
        plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')


    plt.show()


##### PRACENT OF VOTES FOR AECH PARTY
def normlazied_votes(df):
    votes_per_city = df["מצביעים"]
    party_share = df.div(votes_per_city,axis=0)
    return party_share


##### PRACENT OF VOTES FOR AECH PARTY
def full_power_of_voters(df,col):
    full_power = df.mul(col,axis=0)
    return full_power


# Get votes of all parties (normalized)
def parties_votes2(df, thresh):
    par = df.sum().div(df.sum().sum()).sort_values(ascending=False)
    par= par[par > thresh]
    ans=df.sum()
    ans=ans[par.keys()]
    return ans


# Bar plot for all parties with votes above a threshold
def parties_bar_vs_full(df1, df2,thresh):
    width = 0.3
    votes1 = parties_votes2(df1, thresh)  # total votes for each party
    n = len(votes1)  # number of parties
    names = votes1.keys()

    votes2 = parties_votes2(df2, thresh)  # total votes for each party
    n2 = len(votes2)  # number of parties

    rev_names = [name[::-1] for name in list(names)]
    fig, ax = plt.subplots()  # plt.subplots()

    all_bar_real = ax.bar(np.arange(n), list(votes1), width, color='0.1')
    all_bar_full = ax.bar(np.arange(n2)+width, list(votes2), width, color='g')

    ax.set_ylabel('Votes ')
    ax.set_xlabel('Parties Names')
    ax.set_title('Votes per party 2019b real vs full potential')
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(rev_names)
    ax.legend((all_bar_real[0], all_bar_full[0]), ('real', "full potential"))

    # set individual bar lables using above list
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x() , i.get_height() + 12000,
                str(round((i.get_height()))), fontsize=6, color='dimgrey',
                rotation=0)
    plt.show()

    return fig, ax
