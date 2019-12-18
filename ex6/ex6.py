# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:38:48 2019

@author: Liav
"""
# Call libraries

import numpy as np
import pandas as pd
from election_functions_new import *
from sklearn.decomposition import PCA
pd.set_option('display.max_rows', 1300)
pd.set_option('display.max_columns', 1300)

# Path to datafile - change to your directory! (both can be the same directory)
DATA_PATH = r'C:\Users\Liav\Desktop\Uni\Lab\ex6'

#load data

df_sep, df_sep_raw = read_election_results("2019b","ballot")
df_sep_cities, df_sep_cities_raw = read_election_results("2019b","city")

df_apr, df_apr_raw = read_election_results("2019a","ballot")
df_apr_cities, df_apr_cities_raw = read_election_results("2019a","city")


parties_dict ={'אמת' : "עבודה גשר", 'ג' : "יהדות התורה", 'ודעם'  : "הרשימה המשותפת", 'טב'  : "ימינה", 'כף'  : "עוצמה יהודית",
 'ל'  : "ישראל ביתנו", 'מחל'  : "הליכוד", 'מרצ'  : "המחנה הדמוקרטי", 'פה'  : "כחול לבן", 'שס'  : "שס"}

parties_dict_2019a ={'אמת' : "עבודה", 'ג' : "יהדות התורה", 'דעם'  : "רעם בלד", 'ום'  : "חדש תעל", 'טב'  : "איחוד מפלגות הימין",
 'ל'  : "ישראל ביתנו", 'מחל'  : "הליכוד", 'מרצ'  : "מרצ", 'פה'  : "כחול לבן", 'שס'  : "שס",  'כ'  : "כולנו",  'נ'  : "ימין חדש",  'ז'  : "זהות",  'נר'  : "גשר"}


#get 10 biggest parties
big_parties = parties_votes(df_sep,10).index

#convert voting absolute number to voting frequency
p_ballot = df_sep.apply(lambda x: x/x.sum(),axis=1).dropna()

#transpose dataframe for working with PCA
p_ballot_t = np.transpose(p_ballot)

#normalized party vector
p_t_normalized = p_ballot_t.apply(normalize, axis=1).loc[big_parties]
#normalize ballot vector
p_normalized = p_ballot.apply(normalize, axis=1)[big_parties]


# define PCA object
pca_ballot = PCA(n_components=2)
principalComponents = pca_ballot.fit_transform(p_t_normalized)  # fit model. Compute principal components
X_pca = pca_ballot.transform(p_t_normalized)  # Perform PCA transformation
scatter_twoway(X_pca,names=[parties_dict[n] for n in big_parties],label="Parties PCA 2 dimensions, Normalized Vectors ")

# old lab results
p_ballot_t = p_ballot_t.loc[big_parties]
pca_ballot2 = PCA(n_components=2)
principalComponents = pca_ballot2.fit_transform(p_ballot_t)  # fit model. Compute principal components
X_pca2 = pca_ballot2.transform(p_ballot_t)  # Perform PCA transformation
scatter_twoway(X_pca2,names=[parties_dict[n] for n in big_parties],label="Parties PCA 2 dimensions (From Lab 5)")

# define PCA object
pca_ballot3 = PCA(n_components=2)
principalComponents = pca_ballot3.fit_transform(p_normalized)  # fit model. Compute principal components
X_pca3 = pca_ballot3.transform(p_normalized)  # Perform PCA transformation
df = pd.DataFrame(X_pca3)
df.index = p_normalized.index
scatter_twoway_by_two_groups(df,cities=["יהוד-מונוסון","אור יהודה"])

##2
#merge ballots


# Match ballots for 2 elections
b2019a = adapt_df(df_apr_raw, list(parties_dict_2019a.keys()), include_no_vote=True, ballot_number_field_name='מספר קלפי')
b2019b = adapt_df(df_sep_raw, list(parties_dict.keys()), include_no_vote=True, ballot_number_field_name='קלפי')
u = pd.merge(b2019a, b2019b, how='inner', left_index=True, right_index=True)
 # these dataframes contain number of votes for the two elections only in shared ballots
n2019a = b2019a.loc[u.index]
n2019b = b2019b.loc[u.index]



n2019a_p = n2019a.drop(["לא הצביע","ישוב"],axis=1).apply(lambda x: x/x.sum(),axis=1)
n2019b_p = n2019b.drop(["לא הצביע","ישוב"],axis=1).apply(lambda x: x/x.sum(),axis=1)[big_parties]

df_apr_p_fix = fixing_sep_to_apr(n2019a_p, big_parties)

z = ballot_dist_df(n2019b_p,df_apr_p_fix)

names = z.sort_values(ascending=False).head( 10).index

bzb_sep = df_sep_raw.set_index('ballot_id')[["כשרים", "בזב"]]
bzb_apr = df_apr_raw.set_index('ballot_id')[["כשרים", "בזב"]]


ballot_to_city = dict(zip(n2019a.index, n2019a['ישוב']))

subplots_titles = [ballot_to_city[i][::-1]+"_"+i.split("_")[2] for i in names]
xtickslables = [x[::-1] for x in n2019b_p.loc[names].columns]

bzb_apr_lables = bzb_apr.loc[names]
bzb_sep_lables = bzb_sep.loc[names]

bzb_apr_lables.iloc[0,0]

subplot_bars(x=n2019b_p.loc[names], y=df_apr_p_fix.loc[names],
             xticklabels = xtickslables ,title="Top 10 Ballots: Change in votes distribution"
             ,title_str=subplots_titles, texts = z.sort_values(ascending=False).head(10),q=2,
             apr_lables=bzb_apr_lables, sep_lables=bzb_sep_lables)

bzb_apr_lables.iloc[0:]

z = z.sort_values(ascending=False).head(10)
z.index = [x[::-1] for x in subplots_titles]
print(z)




###3 liav
##################################
voted_a = pd.DataFrame({"voted_apr":n2019a.drop(["לא הצביע","ישוב"],axis=1).sum(axis=1),"no_voted_apr":b2019a["לא הצביע"]})
voted_b = pd.DataFrame({"voted_sep":n2019b.drop(["לא הצביע","ישוב"],axis=1).sum(axis=1),"no_voted_sep":b2019b["לא הצביע"]})
u_voted = pd.merge(voted_a, voted_b, how='inner', left_index=True, right_index=True)

u_voted["apr_ratio"] = u_voted["voted_apr"]/(u_voted["voted_apr"]+u_voted["no_voted_apr"])
u_voted["sep_ratio"] = u_voted["voted_sep"]/(u_voted["voted_sep"]+u_voted["no_voted_sep"])
u_voted["avg"] = (u_voted["apr_ratio"]+u_voted["sep_ratio"])/2

top_10_voting_ballot = u_voted["avg"].sort_values(ascending=False).head(10).index



subplots_titles = [ballot_to_city[i][::-1]+"_"+i.split("_")[2] for i in top_10_voting_ballot]
xtickslables = [x[::-1] for x in n2019b_p.loc[names].columns]

# subplot_bars(x=n2019b_p.loc[top_10_voting_ballot], y=df_apr_p_fix.loc[top_10_voting_ballot],
#              xticklabels = n2019b_p.columns ,title="10 Ballots with biggest voting rate."
#              ,title_str=subplots_titles)
##############################################



#David's code for Q.3
# =============================================================================
# ###3
b2019a_3 = adapt_df(df_apr_raw, list(parties_dict_2019a.keys()), include_no_vote=True, ballot_number_field_name='מספר קלפי')
b2019b_3 = adapt_df(df_sep_raw, list(parties_dict.keys()), include_no_vote=True, ballot_number_field_name='קלפי')
u_3 = pd.merge(b2019a_3, b2019b_3, how='inner', left_index=True, right_index=True)
# these dataframes contain number of votes for the two elections only in shared ballots
n2019a_3 = b2019a_3.loc[u.index]
n2019b_3 = b2019b_3.loc[u.index]


###data frame that calculats the votinig pracent
#votinig pracent april
n2019a_vp_3 = n2019a_3.drop(["ישוב"],axis=1).apply(lambda x: (x.sum()- x["לא הצביע"])/x.sum(),axis=1)
#votinig pracent september
n2019b_vp_3 = n2019b_3.drop(["ישוב"],axis=1).apply(lambda x: (x.sum()- x["לא הצביע"])/x.sum(),axis=1)
#votinig pracent avarege
z_3 = (n2019b_vp_3 + n2019a_vp_3)/2
#voting pracent bigest list
names_10_vots_precent = z_3.sort_values(ascending=False).head(10).index

bzb_apr_lables = bzb_apr.loc[names_10_vots_precent]
bzb_sep_lables = bzb_sep.loc[names_10_vots_precent]


#bar plot of voting pracent bigest list
subplot_bars(x=n2019b_p.loc[names_10_vots_precent], y=df_apr_p_fix.loc[names_10_vots_precent],
             xticklabels = [x[::-1] for x in n2019b_p.columns],title="10 Ballots with biggest voting rate",
             title_str=subplots_titles,
             texts = z_3.sort_values(ascending=False).head(10),q=3,
            apr_lables=bzb_apr_lables, sep_lables=bzb_sep_lables)


z_3 = z.sort_values(ascending=False).head(10)
z_3.index = [x[::-1] for x in subplots_titles]
print(z)


# =============================================================================


###4.

subplots_titles = [ballot_to_city[i][::-1]+"_"+i.split("_")[2] for i in top_10_voting_ballot]

#votinig pracent squer dif
z_4 = n2019b_vp_3.sub(n2019a_vp_3)**2
#voting pracent squre dif bigest list
names_10_vots_precent_dif = z_4.sort_values(ascending=False).head(10).index
subplots_titles = [ballot_to_city[i][::-1]+"_"+i.split("_")[2] for i in names_10_vots_precent_dif]


bzb_apr_lables = bzb_apr.loc[names_10_vots_precent_dif]
bzb_sep_lables = bzb_sep.loc[names_10_vots_precent_dif]

#bar plot of voting pracent squre dif bigest list
subplot_bars(x=n2019b_p.loc[names_10_vots_precent_dif], y=df_apr_p_fix.loc[names_10_vots_precent_dif],
             xticklabels = [x[::-1] for x in n2019b_p.columns],title="Top 10 ballots: Voting Precentage difference",
             title_str=subplots_titles
             ,q=4,texts = z_4.sort_values(ascending=False).head(10),
             apr_lables=bzb_apr_lables, sep_lables=bzb_sep_lables)


z_4 = z.sort_values(ascending=False).head(10)
z_4.index = [x[::-1] for x in subplots_titles]
print(z)

###5.
####voting destrabution no left partys april
df_apr_p_fix_no_left=df_apr_p_fix[['טב','מחל', 'שס', 'ג', 'כף']]
####voting destrabution no left partys september
n2019b_p_no_left=n2019b_p[['טב','מחל', 'שס', 'ג', 'כף']]
#votinig pracent squer dif of no left partys
z_5 = ballot_dist_df(n2019b_p_no_left,df_apr_p_fix_no_left)
#voting pracent squre dif list no left
names_10_no_left_dif = z_5.sort_values(ascending=False).head(10).index
subplots_titles = [ballot_to_city[i][::-1]+"_"+i.split("_")[2] for i in names_10_no_left_dif]


bzb_apr_lables = bzb_apr.loc[names_10_no_left_dif]
bzb_sep_lables = bzb_sep.loc[names_10_no_left_dif]

#bar plot of voting pracent squre dif bigest list
subplot_bars(x=n2019b_p.loc[names_10_no_left_dif], y=df_apr_p_fix.loc[names_10_no_left_dif],
             xticklabels = [x[::-1] for x in n2019b_p.columns],title="Top 10 Ballots: Biggest change (right parties)",
             title_str=subplots_titles,q=5,
             texts = z_5.sort_values(ascending=False).head(10),
             apr_lables=bzb_apr_lables, sep_lables=bzb_sep_lables)


z_5 = z.sort_values(ascending=False).head(10)
z_5.index = [x[::-1] for x in subplots_titles]
print(z)

###############
#BONUS
###############

#BONUS
l = list((df_apr_raw.set_index("ballot_id")["כשרים"]/df_apr_raw.set_index("ballot_id")["מצביעים"]).sort_values().head(10).index)


subplots_titles = [ballot_to_city[i][::-1]+"_"+i.split("_")[2] for i in l]

#bar plot of voting pracent squre dif bigest list
subplot_bars(x=n2019b_p.loc[l], y=df_apr_p_fix.loc[l],
             xticklabels = [x[::-1] for x in n2019b_p.columns],title="Top 10 Ballots: Biggest change (right parties)",
             title_str=subplots_titles,q=5,
             texts = z_5.sort_values(ascending=False).head(10),
             apr_lables=bzb_apr_lables, sep_lables=bzb_sep_lables)
