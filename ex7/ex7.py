# -*- coding: utf-8 -*-
"""
@author: Liav
"""
# Call libraries
from election_functions_new import *
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Path to datafile - change to your directory! (both can be the same directory)
DATA_PATH = r'C:\Users\Liav\Desktop\Uni\Lab\ex7'

#load raw data, and cut metadata
df_sep, df_sep_raw = read_election_results("2019b","ballot")
df_apr, df_apr_raw = read_election_results("2019a","ballot")

# create parties-vote notes dictionary
parties_dict_sep ={'אמת' : "עבודה גשר", 'ג' : "יהדות התורה", 'ודעם'  : "הרשימה המשותפת", 'טב'  : "ימינה", 'כף'  : "עוצמה יהודית",
 'ל'  : "ישראל ביתנו", 'מחל'  : "הליכוד", 'מרצ'  : "המחנה הדמוקרטי", 'פה'  : "כחול לבן", 'שס'  : "שס"}
parties_dict_apr ={'אמת' : "עבודה", 'ג' : "יהדות התורה", 'דעם'  : "רעם בלד", 'ום'  : "חדש תעל", 'טב'  : "איחוד מפלגות הימין",
 'ל'  : "ישראל ביתנו", 'מחל'  : "הליכוד", 'מרצ'  : "מרצ", 'פה'  : "כחול לבן", 'שס'  : "שס",  'כ'  : "כולנו",  'נ'  : "ימין חדש",  'ז'  : "זהות",  'נר'  : "גשר"}

names_a = parties_votes(df_apr, 14).index
names_b = parties_votes(df_sep,10).index


# Match ballots for 2 elections
df_apr = adapt_df(df_apr_raw, names_a, include_no_vote=True, ballot_number_field_name='מספר קלפי')
df_sep = adapt_df(df_sep_raw, names_b, include_no_vote=True, ballot_number_field_name='קלפי')  # Choose if to include non-voters



u = pd.merge(df_apr, df_sep, how='inner', left_index=True, right_index=True)

v_2019a = df_apr.loc[u.index].drop(["לא הצביע","ישוב"],axis=1)
v_2019b = df_sep.loc[u.index].drop(["לא הצביע","ישוב"],axis=1)

#convert to matrix
v1 = v_2019a.values
v2 = v_2019b.values

#Q 1. Linear regression with thresholding (only parties)
#a - non normalized
#### method 1: closed-form solution .
# Least-squares Formula: M-hat = [v1^T * v1]^(-1) * v1^T * v2
M  = np.linalg.pinv(v1.T @ v1) @ v1.T @ v2

# M = np.linalg.pinv(v1.T @ v1) @ v1.T @ v2
heatmap(M,v_2019b.columns,v_2019a.columns)

# b - normalized
np.place(M, M< 0.5/100, 0)
z = M/M.sum(axis=1)[:,None]
heatmap(z,v_2019b.columns,v_2019a.columns)


###################
# Q2 - same as Q1 but with non voters

v1_4 = df_apr.loc[u.index].drop(["ישוב"],axis=1).values
v2_4 = df_sep.loc[u.index].drop(["ישוב"],axis=1).values
#a - non normalized
M_4 = np.linalg.pinv(v1_4.T @ v1_4) @ v1_4.T @ v2_4
heatmap(M_4,v_2019b.columns,v_2019a.columns,True)


# b - normalized
np.place(M_4, M_4< 0.5/100, 0)
z = M_4/M_4.sum(axis=1)[:,None]
heatmap(z,v_2019b.columns,v_2019a.columns,True)

#Q3
# method 2: non-negative least square solution
# Solve argmin_x || Ax - b ||_2 for x>=0.
M = np.zeros((v2.shape[1], v1.shape[1]))
for i in range(v2.shape[1]):
    sol, r2 = nnls(v1, v2[:, i])
    M[i,:] = sol
    pred = v1 @ sol

M = M.T
heatmap(M,v_2019b.columns,v_2019a.columns)

# resid = Y-Y_hat
# Y_hat = X@(X.T @ X)(-1) @ X.T @ Y
Y_hat = v1_4 @ M_4
resid = abs(v2_4 - Y_hat)

print("!!!!!!!!!!!")
print(resid.shape)

resid_df = pd.DataFrame(resid.mean(axis=0))
print(resid_df)
resid_df.columns = ["Residuals"]
ax = resid_df.plot.bar(y='Residuals')
ax.set_xticklabels([x[::-1] for x in v_2019b.columns]+["עיבצה אל"], rotation=90)
ax.set_title("Residuals Averages for Parties")
plt.show()

vote_movements = M.T * v_2019a.sum(axis=0).values
sankey_plot(vote_movements, [x[::-1] for x in v_2019a.columns], [x[::-1] for x in v_2019b.columns])
