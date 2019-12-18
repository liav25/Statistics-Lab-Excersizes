# Estimate and correct for voting turnout
# Modified from a python notebook from Harel Kein
# Call libraries
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import stats
import statsmodels.api as sm

pd.set_option('display.max_rows', 1300)
pd.set_option('display.max_columns', 1300)

#np.random.seed(23)

from election_functions import *

# Path to datafile - change to your directory!
DATA_PATH = r'C:\Users\Liav\Desktop\Uni\Lab\ex3'

# Correct for voting turnout in cities/ballots (from lab2)
# df - a voting distirbution data frame
# v - precent of voters on each kalpi (ahuz ksherim)
# return:  q_hat - a matrix of 100% voting of bzb (modereated to integers)
#         q - normalized dist. in case all the citizens were voting
def n_tilde_generator(df, v):
    """
    n_tilde generator =: n_tilde is a df represent hypothetical 100% voting
    :param df: votes dataframe
    :param v: vector of vote percentage (ksherim/bzb)
    :return: n_tilde
    """
    q_hat = df.div(v, axis='rows')


    return q_hat.fillna(0).round(0).astype(int)
#
def simple_turnout_correction(df, v):
    """
    a function that build a simple turnout from lab 02
    :param df: votes dataframe
    :param v: vector of vote percentage (ksherim/bzb)
    :return: a vector of bechirot results after turnout correction
    """
    # v =: precent of voters in each kalpi
    # q_hat := number of votes on kapli / precent of kolot ksherim
    q_hat = df.div(v, axis='rows')
    # q_hat := new distribution of parties after "full"
    q_hat = q_hat.sum().div(q_hat.sum().sum())  # Simple correction

    return q_hat

def rev(lst_str):
    """
    :param lst_str: list of strings
    :return: reversed string for all list
    """
    return [x[::-1] for x in lst_str]


def bechirot_simulation(n_tilde, pij):
    """
    # this function get a vector of natural numbers and a proportion (or vector of proportions)
    # and generate a random binomial simulation
    :param n_tilde: a vector represent the number of experiments
    :param pij:
    :return:  a scalar or a vector of proportions
    """
    sim_df = pd.DataFrame(np.random.binomial(n=n_tilde, p=pij))
    sim_df.index = n_tilde.index
    sim_df.columns = n_tilde.columns
    return sim_df

def get_means(d):
    """
    :param d: a dictionary with all simulations results
    :return: list of average results of all simulations (per party)
    """
    return [np.mean(x) for x in d.values()]
def get_sd(d):
    """
    :param d: a dictionary with all simulations results
    :return: list of standard variation of the simulations result (per party)
    """
    return [np.std(x) for x in d.values()]


def turnout_bar(p, q, q2, labels):
    """
    :param p: vector of Real Results (for example "po":0.25, "mahal":0.24..)
    :param q: vector of simulation results
    :param q2: vector of simulation results after turnout correction
    :param labels: the labels of legend
    :return: plot
    """
    width = 0.2
    names = n_tilde.columns
    rev_names = [name[::-1] for name in list(names)]
    fig, ax = plt.subplots()  # plt.subplots()

    n1 = len(p)
    orig_bar = ax.bar(np.arange(n1), p, width, color='b',)
    adj_bar = ax.bar(np.arange(n1)+width, q[0], width, color='r', yerr=q[1], capsize=4)
    adj_bar2 = ax.bar(np.arange(n1)+2*width, q2[0], width, color='g', yerr=q2[1], capsize=4)

    ax.set_ylabel('Votes percent')
    ax.set_xlabel('Parties Names')
    ax.set_title('Votes percent per party 2019 with/without turnout adjustment')
    ax.set_xticks(np.arange(n1))
    ax.set_xticklabels(rev_names)
    ax.legend((orig_bar[0], adj_bar[0], adj_bar2[0]), labels)
    plt.show()

    return fig, ax

############################################
# Data analysis commands below
############################################

analysis = 'ballot' # city # choose if to analyze by city or ballot
#load  data
df_sep_raw = pd.read_csv(os.path.join(DATA_PATH, r'votes per '+analysis+' 2019b.csv'), encoding = 'iso-8859-8', index_col='שם ישוב').sort_index()
#drop semel
df_sep = df_sep_raw.drop('סמל ועדה', axis=1)  # new column added in Sep 2019
#drop hizoniot
df_sep = df_sep[df_sep.index != 'מעטפות חיצוניות']
if analysis == 'city':
    first_col = 5
else:
    first_col = 9

df_sep = df_sep[df_sep.columns[first_col:]]  # removing "metadata" columns

#add voting precent
df_sep_raw2 = df_sep_raw[df_sep_raw.index != 'מעטפות חיצוניות'] # remove

# Compute p_ij two times: (use numpy 'tile' command to create a matrix from a vector)
# 1. from v_i: ...
# v =: precent of voters in each kalpi
v = df_sep_raw2['כשרים'] / df_sep_raw2['בזב']
#fix values that above 1
v[v>1] = 1
#get top ten parties names
top_ten = list(((df_sep.sum()/df_sep.sum().sum()).sort_values(ascending=False).head(10)).index)
#select only top ten parties
df_sep = df_sep[top_ten]
#create a turnout - what if everybody vote ?
n_tilde = n_tilde_generator(df_sep, v)

#duplicate v to be from correct dimension for the random binomial
pij = np.tile(np.transpose( np.array([v])),(1,10))

#create a vector for voting precent by size of party
#we decided by what we think from our knowledge
lst = [0.8,0.85,0.5,0.9,0.7,0.95,0.75,0.6,0.65,0.55]
#fix dimension for binomial random
alpha = np.tile(lst,(n_tilde.shape[0],1))


bzb = df_sep_raw2['בזב']
Y = bzb
X = df_sep
# X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
#result 2
alpha_j = results.params
alpha_ols = np.tile(alpha_j,(n_tilde.shape[0],1))

z=(((1/alpha_j)*df_sep).round().astype(int))
z_pres = ((z.sum().div(z.sum().sum())) ,0)

dict_a = {}
dict_a_correct = {}
dict_b = {}
dict_b_correct = {}
dict_c = {}


for col in n_tilde.columns:
    dict_a[col] = []
    dict_b[col] = []
    dict_a_correct[col] = []
    dict_b_correct[col] = []

    
ITER = 25

for i in range(ITER):
    # print(i,end="")
    sim_a = bechirot_simulation(n_tilde, pij)
    sim_b = bechirot_simulation(n_tilde, alpha)
    for name in sim_a:
        dict_a[name] += [sim_a.sum().div(sim_a.sum().sum())[name]]
        dict_b[name] += [sim_b.sum().div(sim_b.sum().sum())[name]]
        dict_a_correct[name] += [simple_turnout_correction(sim_a,v)[name]]
        dict_b_correct[name] += [simple_turnout_correction(sim_b,v)[name]]



sim_a = (get_means(dict_a), get_sd(dict_a))
sim_a_cor = (get_means(dict_a_correct),get_sd(dict_a_correct))
sim_b = (get_means(dict_b), get_sd(dict_b))
sim_b_cor = (get_means(dict_b_correct),get_sd(dict_b_correct))
real_results = df_sep.sum().div(df_sep.sum().sum())

#Q1
turnout_bar(real_results,sim_a,sim_a_cor,rev(['תוצאות אמת','סימולציה א','סימולציה א - תיקון']))
turnout_bar(real_results,sim_b,sim_b_cor,rev(['תוצאות אמת','סימולציה ב','סימולציה ב - תיקון']))

#Q2
turnout_bar(real_results,sim_a,(sim_a[0]*(1/alpha_j),sim_a[1]),rev(['תוצאת אמת','סימולציה א','תוצאות סימולציה א - תיקון רגרסיה']))
turnout_bar(real_results,sim_b,(sim_b[0]*(1/alpha_j),sim_b[1]),rev(['תוצאת אמת','סימולציה ב','תוצאות סימולציה ב - תיקון רגרסיה']))
#Q3
turnout_bar(real_results,z_pres,(n_tilde.sum().div(n_tilde.sum().sum()),0),rev(['תוצאות אמת','תוצאות אמת - תיקון רגרסיה','תוצאות אמת - תיקון רגיל']))

