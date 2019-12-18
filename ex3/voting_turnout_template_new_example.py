# Estimate and correct for voting turnout
# Modified from a python notebook from Harel Kein
# Call libraries
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import stats
import statsmodels.api as sm

from election_functions import *

# Path to datafile - change to your directory!
DATA_PATH = 'C:/Users/Or Zuk/Google Drive/HUJI/Teaching/Lab_52568/Data/Elections'


# Correct for voting turnout in cities/ballots (from lab2)
def simple_turnout_correction(df, v):

    p = df.sum().div(df.sum().sum())  # votes without correction
    q_hat = df.div(v, axis='rows')
    q_hat = q_hat.sum().div(q_hat.sum().sum())  # Simple correction

    return p, q_hat


# Correct for voting turnout in cities
def regression_turnout_correction(df, v):

    p = df.sum().div(df.sum().sum())  # votes without correction
    # bzb = ... compute or read from dataframe
    # least squares correction. Use OLS function of statsmodel:
    #  model = sm.OLS(y, X).fit()
    # where you have to determine the explained vector y and the design matrix X
    #
    # compute q-hat using estimated regression coefficients alpha_j^(-1)
    # q_hat = ...

    return p, q_hat


# Bar plot for all parties with votes above a threshold for 3 different bars
def turnout_bar(p, q, q2, thresh, labels):
    width = 0.2
    names = p[p>thresh].keys()
    rev_names = [name[::-1] for name in list(names)]
    fig, ax = plt.subplots()  # plt.subplots()

    p1 = p[p>thresh]
    q1 = q[p>thresh]
    q2 = q2[p>thresh]
    n1 = len(p1)
    orig_bar = ax.bar(np.arange(n1), list(p1), width, color='b')
    adj_bar = ax.bar(np.arange(n1)+width, list(q1), width, color='r')
    adj_bar2 = ax.bar(np.arange(n1)+2*width, list(q2), width, color='g')

    ax.set_ylabel('Votes percent')
    ax.set_xlabel('Parties Names')
    ax.set_title('Votes percent per party 2019 with/without turnout adjustment')
    ax.set_xticks(np.arange(n1))
    ax.set_xticklabels(rev_names)
    ax.legend((orig_bar[0], adj_bar[0], adj_bar2[0]), labels)
    plt.show()

    return fig, ax

# Simulate
def sample_turnout(df, p_mat):

    # Sample votes for each ballot and party. Use the function random.binomial from numpy
    df_vote = ... # complete code
    # For example, the following code samples 2*4 binomial variables with different values of n and p
    # k = np.random.binomial([[10, 20, 30], [1, 2, 3]], [[0.1, 0.9, 0.5], [0.8, 0.6, 0.4]], (2, 3))

    return df_vote


############################################
# Data analysis commands below
############################################

analysis = 'ballot' # city # choose if to analyze by city or ballot

df_sep_raw = pd.read_csv(os.path.join(DATA_PATH, r'votes per '+analysis+' 2019b.csv'), encoding = 'iso-8859-8', index_col='שם ישוב').sort_index()
df_sep = df_sep_raw.drop('סמל ועדה', axis=1)  # new column added in Sep 2019
df_sep = df_sep[df_sep.index != 'מעטפות חיצוניות']
if analysis == 'city':
    first_col = 5
else:
    first_col = 9
df_sep = df_sep[df_sep.columns[first_col:]]  # removing "metadata" columns

df_sep_raw2 = df_sep_raw[df_sep_raw.index != 'מעטפות חיצוניות'] # remove
v = df_sep_raw2['כשרים'] / df_sep_raw2['בזב']






# Simulation study:
# Here compute n_ij-tilde and q

# Compute p_ij two times: (use numpy 'tile' command to create a matrix from a vector)
# 1. from v_i: ...
# v = ...
# p_mat = ...
# For example, the following creates a 2*3 numpy array from a vector of 3:
# p_mat = np.tile([0.7, 0.6, 0.8], (2,1))
# 2. from alpha_j:
# alpha = ...
# p_mat = ...
# Update mean and variance of p_j, q_hat_j for each party j

# Here perform simulations
iters = 25
for iter in range(iters):
    # Simulate data
    # ... = sample_turnout(df, p_mat)


    # Apply correction and estimate q:
    # p,q_corr = simple_turnout_correction(...)
    # p,q_reg_corr = regression_turnout_correction(...)

    # Update mean and st.d. of observed votes p and estimators q,q_reg
    # ...



# Plot p, q and q_reg in a bar-plot for each option:
# turnout_bar(p, q, q_corr, 0.005, ('q', 'p', 'q-hat'))  # show


# Apply to corrections to the actual data
p,q_corr = simple_turnout_correction(df_sep, v)
p, q_corr_reg = regression_turnout_correction(df_sep, v)

# Plot results