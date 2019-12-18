# Load and analysis of election data
# Modified from a python notebook from Harel Kein
# Call libraries

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (10.0, 5.0)

#show all rows and columns while printing
pd.set_option('display.max_rows', 5100)
#pd.set_option('display.max_columns', 500)


# Path to datafile - change to your directory!
DATA_PATH = r'C:\Users\Liav\Desktop\Uni\Lab\ex1'


# Functions
# Get number of votes of all parties
def parties_votes_total(df, thresh):
    par = df.sum().sort_values(ascending=False)
    return par[par > thresh]

#get the distance of two multiniomial distributions
def multinomial_distance(vec1, vec2):
    res = 0
    for i in range(len(vec1)):
        res += (vec1[i]-vec2[i])**2
    return res

#get the normalized voting rate
def normalized_votes(df, parties):
   return df[parties].div(df['מצביעים'], axis=0)

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


#plot histogram of specific column
def series_histogram(series, nbins, xlab, ylab):
    plt.hist(series, nbins, rwidth=0.8)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title('Histogram of ' + str(series.name))
    plt.show()
    return plt

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

    plt.scatter(party_share1, party_share2)
    plt.xlabel(party1[::-1])
    plt.ylabel(party2[::-1])
    plt.title('Scatter for two parties ' )
    plt.show()

def parties_barplot_two_cities(df, thresh, city1, city2):
    width = 0.3
    votes = parties_votes(df, thresh)  # total votes for each party
    n = len(votes)  # number of parties
    names = votes.keys()

    rev_names = [name[::-1] for name in list(names)]
    fig, ax = plt.subplots()  # plt.subplots()

    city1_votes = df.loc[city1, names] / df.loc[city1,names].sum()
    city2_votes = df.loc[city2, names] / df.loc[city2,names].sum()
    city_bar1 = ax.bar(np.arange(n), list(city1_votes), width, color='b')
    city_bar2 = ax.bar(np.arange(n)+width, list(city2_votes), width, color='r')

    ax.set_ylabel('Votes percent')
    ax.set_xlabel('Parties Names')
    ax.set_title('Votes percent per party 2019')
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(rev_names)
    ax.legend((city_bar1[0], city_bar2[0]), (city1[::-1], city2[::-1]))


    plt.show()

    return fig, ax
############################################
# Data analysis commands below
############################################

# 1. Read data
df_sep_raw = pd.read_csv(os.path.join(DATA_PATH, r'votes per city 2019b.csv'), encoding = 'iso-8859-8', index_col='שם ישוב').sort_index()
df_sep = df_sep_raw.drop('סמל ועדה', axis=1) # new column added in Sep 2019
df_sep_no_meta = df_sep[df_sep.columns[5:]]

# Question 1. Histogram of kolot psulim percentage
# create a disqualified votes percentage columns
df_sep['disqualified_votes_per'] = (df_sep['פסולים']/df_sep['מצביעים'])*100

# show histogram
plt = series_histogram(df_sep["disqualified_votes_per"], 100, 'Percent of disqualified Votes','Number of Cities')


#finding the city with maximum disqualifed votes percentage
row_idx = df_sep[['disqualified_votes_per']].idxmax()
ans = df_sep.loc[row_idx]['disqualified_votes_per']

print("City with highest disqualified votes percentage is {} with {}".format(row_idx.values[0], round(ans[0],3))
      + " percent\n")

# Q2. show 3 city couples bar plot

#print list of biggest cities (above 50,000 votes)
big_cities = (list(df_sep[df_sep['מצביעים']>50000].index))

fig , ax= parties_barplot_two_cities(df_sep_no_meta, 0.0325,"חולון", "בת ים")
fig.savefig(os.path.join(DATA_PATH,"cities barplot Bat Yam vs. Holon.png"))

fig, ax = parties_barplot_two_cities(df_sep_no_meta, 0.0325,"אשדוד", "נתניה")
fig.savefig(os.path.join(DATA_PATH,"cities barplot  Ashdod vs. Netanya.png"))

fig, ax = parties_barplot_two_cities(df_sep_no_meta, 0.0325,"באר שבע", "חיפה")
fig.savefig(os.path.join(DATA_PATH,"cities barplot Beer Sheva vs. Haifa.png"))


# Q3. cities vs. Israel distribution
# get the votes normalization
normalized_votes_df = normalized_votes(df_sep, list(df_sep_no_meta.columns))

# get country votes normalization
israel_votes = normalized_votes(df_sep.sum(axis=0),list(df_sep_no_meta.columns))

# find multinomial distance for each city (row)
distance_vector = normalized_votes_df.apply(multinomial_distance,axis=1,args=[israel_votes])

# n closest cities
n=2
n_smallest = distance_vector.nsmallest(n)
most_unsimilar = (distance_vector.idxmax(), distance_vector[distance_vector.idxmax()])

#save fig and print data
print("The most similar city to country distribution is {} with {} distance".format(n_smallest.index[0],n_smallest[0]))
print("The 2nd most similar city to country distribution is {} with {} distance".format(n_smallest.index[1],n_smallest[1]))

#save fig and print data
fig, ax = parties_bar(df_sep_no_meta,0,n_smallest.index[0])
fig.savefig(os.path.join(DATA_PATH,"first_closest.png"))

#save fig and print data
fig, ax = parties_bar(df_sep_no_meta,0,n_smallest.index[1])
fig.savefig(os.path.join(DATA_PATH,"secong_closest.png"))

#save fig and print data
print("The most unsimilar city to country distribution is {} with {} distance".format(most_unsimilar[0],most_unsimilar[1]))
fig, ax =parties_bar(df_sep_no_meta,0,most_unsimilar[0])
fig.savefig(os.path.join(DATA_PATH,"most_unsimilar.png"))
