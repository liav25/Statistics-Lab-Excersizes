# Functions for Loading and analysis of election data
# Modified from a python notebook from Harel Kein
# Call libraries
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
pd.set_option('display.max_rows', 1300)
pd.set_option('display.max_columns', 1300)

DATA_PATH = r'C:\Users\Liav\Desktop\Uni\Lab\ex6'

# Functions
def parties_votes_total(df, n):
    """
    # Get number of votes of all parties
    :param df: a vote df
    :param n: the number of parties to show
    :return: top n parties
    """
    par = df.sum().sort_values(ascending=False)
    return par.head(n)


def parties_votes(df, n):
    """
    # Get votes of all parties (precentege)
    :param df: a vote df
    :param n: the number of parties to show
    :return: top n parties votes precentage
    """
    par = df.sum().div(df.sum().sum()).sort_values(ascending=False)
    return par.head(n)



########################################david election results for df of votes number and thresh
def parties_bar_reg_pracent(df1,n):
    """
    :param df1: a vote df
    :param n: the top n parties    
    :return:
    """
    width = 0.3
    votes1 = parties_votes(df1, n)  # total votes for each party
    n = len(votes1)  # number of parties
    names = votes1.keys()

    rev_names = [name[::-1] for name in list(names)]
    fig, ax = plt.subplots()  # plt.subplots()
    all_bar_real = ax.bar(np.arange(n), list(votes1), width, color='dimgrey')

    ax.set_ylabel('Votes pracent')
    ax.set_xlabel('Parties Names')
    ax.set_title('Votes precent per party 2019b')
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(rev_names)

    # set individual bar lables using above list
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x() , i.get_height() + 0.001,
                str(round((i.get_height())*100,ndigits=2)), fontsize=8, color='black',
                rotation=0)
    plt.show()

    return fig, ax


########################################david election results for df of votes number and thresh
def parties_bar_double(df1,df2,n,labels):
    """
    :param df1: a vote df
    :param df2: another vote df
    :param thresh: the threshold for minimum ahuz hasima    :return:
    """
    width = 0.3
    votes1 = parties_votes(df1, n)  # total votes for each party
    votes2 = parties_votes(df2, n)
    names = votes1.keys()

    n = len(votes1)  # number of parties
    X = np.arange(n)
    
    rev_names = [name[::-1] for name in list(names)]

    fig, ax = plt.subplots()  # plt.subplots()
     
    a = ax.bar(X, list(votes1), width, color='#000080',label=labels[0])
    b = ax.bar(X+width, list(votes2), width, color='#6593F5',label=labels[1])
    
    ax.set_ylabel('Votes pracent')
    ax.set_xlabel('Parties Names')
    ax.set_title('Votes precent per party 2019b')
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(rev_names)
    
    plt.grid(True,'major','y', ls='--',lw=.5,c='k',alpha=.2)
    ax.legend()
    plt.show()
    fig.tight_layout()

    return fig, ax


########################################david election results for df of votes number and thresh
def parties_bar_by_rows(df1):
       
    width = 0.3
    for index, row in df1.iterrows():

        votes1 = row
        names=(pd.DataFrame(votes1).index)
        rev_names = [name[::-1] for name in list(names)]
        n = len(votes1)
        fig, ax = plt.subplots()  # plt.subplots()
        all_bar_real = ax.bar(np.arange(n), list(votes1), width, color='dimgrey')

        ax.set_ylabel('Votes pracent')
        ax.set_xlabel('Parties Names')
        ax.set_title('Votes precent per party 2019b in the : pracentile')
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(rev_names)

        # set individual bar lables using above list
        for i in ax.patches:
            # get_x pulls left or right; get_height pushes up or down
            ax.text(i.get_x() , i.get_height() + 0.001,
                    str(round((i.get_height())*100,ndigits=2)), fontsize=8, color='black',
                    rotation=0)
        plt.show()


def  parties_bar_multible(df1):

    # Bar Plot votes for each eshkol in a different subplot. Use plt.subplots
    width = 0.5

    fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(16, 8))
    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]
    n = len(axs)
    names = list((pd.DataFrame(df1.columns))[0])
    rev_names = [name[::-1] for name in list(names)]
    eskol = ['אשכול 1'[::-1],'אשכול 2'[::-1],'אשכול 3'[::-1],'אשכול 4'[::-1],'אשכול 5'[::-1],'אשכול 6'[::-1],'אשכול 7'[::-1],'אשכול 8'[::-1],'אשכול 9'[::-1],'אשכול 01'[::-1]]
    bar1 = []
    for i in range(len(axs)):
        bar1.append(axs[i].bar(np.arange(n), list(df1.loc[i]), width, color='navy'))
        axs[i].legend("")
        axs[i].set_ylabel('Votes pracent')
        axs[i].set_title(eskol[i])
        axs[i].set_xticks(np.arange(n))
        axs[i].set_xticklabels(rev_names,rotation=90)

        # set individual bar lables using above list
        for j in axs[i].patches:
            # get_x pulls left or right; get_height pushes up or down
            axs[i].text(j.get_x(), j.get_height() + 0.001,
                    str(round((j.get_height()) * 100, ndigits=1)), fontsize=6, color='black',
                    rotation=0)

    plt.tight_layout()
    plt.show()


def heterogini_scatter_plot(df):

    cm = plt.cm.get_cmap('viridis_r')

    scat = plt.scatter(x=df['dist'], y=df['gini'],
                       c=np.log(df['pop']), cmap=cm,alpha=.7, s=(df['kmr'])/2)

    plt.colorbar(scat)
    plt.title("Israel Town Heterogeneity vs. GINI")
    plt.xlabel("Heterogeneity")
    plt.ylabel("GINI")
    plt.legend(scatterpoints=1,loc='upper right',  ncol=1, fontsize=8,title="Size = population/kmr\nColor: Log(Population)")
    plt.show()


def distance_heteroginity(shared_cities_df, p_ballot, p_city, df_hev_shared):
    ind = shared_cities_df.index
    final = pd.DataFrame(np.zeros((len(ind), 1)))
    final = final.set_index(ind)

    for i in range(len(ind)):
        ballot_j = p_ballot.loc[ind[i]]
        ballot_city_dist = (ballot_j - p_city.loc[ind[i]])
        dist_transpose = np.transpose(ballot_city_dist)
        ans = (ballot_city_dist.as_matrix() @ dist_transpose.as_matrix())

        # if result matrix is 1x1 dimension
        if ans.shape != ():
            ans = ans.diagonal()
            ans = sum(ans) / len([ans])

        final.loc[ind[i]] = ans

    gini = df_hev_shared["מדד ג'יני[2]"]
    population = df_hev_shared["אוכלוסייה[1]"].str.replace('\W', '').astype(int)
    kmr = df_hev_shared['דירוג.2'].str.replace('\W', '').astype(int)
    final_df = pd.concat([final, gini, population, kmr], axis=1)
    final_df.columns = ['dist', 'gini', 'pop', 'kmr']
    return final_df


# =============================================================================
# def scatter_twoway( df ,names="",s=0.01,label=""):
#     x = df[:, 0]
#     y = df[:, 1]
#     fig, ax = plt.subplots()
#     plt.scatter(x, y,s=s*2000, alpha=.7)
#     plt.xlabel('PC1')
#     plt.ylabel('PC2')
#     plt.title(label)
#     
#     for i in range(len(names)):
#         plt.text(list(x)[i]-0.15, list(y)[i]+0.2, str(names[i][::-1]))
#     plt.show()
#     
#     return plt
# =============================================================================

def scatter_twoway(df1, df2=[],names="",s1=0.01,s2=0.01,label="",arrows=[],longest=[]):

    x1 = df1[:, 0]
    y1 = df1[:, 1]
    fig, ax = plt.subplots(figsize=(25, 10))
    ax.scatter(x1, y1,s=s1*2000, alpha=.3,c='b')
    if len(df2)>1:
        x2 = df2[:, 0]
        y2 = df2[:, 1]
        ax.scatter(x2, y2,s=s2*2000, alpha=.3,c='r')
    if len(arrows)>10:
        for i in arrows:
            if i in longest:
                ax.arrow(x1[i],y1[i],x2[i]-x1[i],y2[i]-y1[i], shape="full", color='green',alpha=1
                         ,width=0.003)
            else:
            #plt.arrow(d['X0'],d['Y0'],d['X1']-d['X0'], d['Y1']-d['Y0'], 
                ax.arrow(x1[i],y1[i],x2[i]-x1[i],y2[i]-y1[i], shape="full", color='black',alpha=0.15)
            #ax.arrow(0, 0, 0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')
            
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.title(label)

    for i in range(len(names)):
        plt.text(list(x1)[i], list(y1)[i], str(names[i][::-1]))
    plt.show()

    return plt


def scatter_twoway_by_two_groups(df,cities):
    g1, g2 = cities[0], cities[1]
    x = df.loc[:,0]
    y = df.loc[:,1]
    x1=x.loc[g1]
    y1=y.loc[g1]
    x2=x.loc[g2]
    y2=y.loc[g2]

    fig, ax = plt.subplots(figsize=(25, 10))
    ax.scatter(x, y, color="grey", alpha=0.1)
    ax.scatter(x1, y1 , color="red", alpha=.7, label=g1[::-1])
    ax.scatter(x2, y2 , color="green", alpha=.7, label=g2[::-1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("Ballots PCA 2 dimensions, Normalized Vectors")
    plt.legend()

    plt.show()
    
    return fig, ax


def fixing_sep_to_apr(score_vec, output):
    HALF = .5
    ihud_yamin = score_vec["טב"]
    yamin_hadash = score_vec["נ"]
    zehut = score_vec["ז"]
    kulanu = score_vec["כ"]
    gesher = score_vec["נר"]
    likud = score_vec["מחל"]
    avoda = score_vec["אמת"]
    
    score_vec["ודעם"] = score_vec["דעם"] + score_vec["ום"] #meshutefet
    score_vec["כף"] = ihud_yamin * HALF  # otzma yehudit
    score_vec["טב"] = yamin_hadash + HALF*(zehut+ihud_yamin)
    score_vec["מחל"] = likud + kulanu + HALF*zehut
    score_vec["אמת"] = avoda + gesher

    return score_vec[output]


def fixing_sep_to_apr2(score_vec):
    ZERO = 0
    HALF=.5




    score_vec["כף"] = score_vec["טב"] *HALF #otzma yehudit
    score_vec["ודעם"] = ZERO
    zehut_yamin = score_vec["ז"] + score_vec["טב"]
    for idx, val in score_vec.iteritems():
        if idx=="ום": #hadash_taal + raam balad
            score_vec["ודעם"] = score_vec["דעם"]+val 
            score_vec.drop(idx,inplace=True)
            score_vec.drop("דעם",inplace=True)

        elif idx=="ז": #zehut
            score_vec["מחל"] = score_vec["מחל"] + val*HALF
            score_vec.drop(idx,inplace=True)
        elif idx=="כ": #KULANU
            score_vec["מחל"] = score_vec["מחל"]+val 
            score_vec.drop(idx,inplace=True)
        elif idx=="נר": #Orly Levi
            score_vec["אמת"] = score_vec["אמת"]+val 
            score_vec.drop(idx,inplace=True)
        elif idx=="טב":
            score_vec["טב"] = score_vec["נ"] + zehut_yamin*HALF #yamina
            score_vec.drop('נ',inplace=True)
    return score_vec.sort_values(ascending=False).sort_index()

def calc_dist(x1,y1,x2,y2):  
     dist = ((x2 - x1)**2 + (y2 - y1)**2)
     return dist  
 
def ballot_dist(kalpi_a,kalpi_b):
    dist =  (np.subtract(kalpi_a,kalpi_b))**2
    return sum(dist)

def ballot_dist_df(df1, df2):
    c = df1.sub(df2)**2
    return c.sum(axis=1)


def normalize(v):     
    norm = np.linalg.norm(v)     
    if norm == 0: 
        return v 
    return v / norm


# Read election results from csv file
def read_election_results(year, analysis):
    df_raw = pd.read_csv(os.path.join(DATA_PATH, r'votes per ' + analysis + ' ' + year + '.csv'),
                             encoding='iso-8859-8', index_col='שם ישוב').sort_index()
    if year == '2019b':
        df = df_raw.drop('סמל ועדה', axis=1)  # new column added in Sep 2019
    else:
        df = df_raw

    df = df[df.index != 'מעטפות חיצוניות']
    
    if analysis == 'city':
        first_col = 5
    else:  # 
        if year=="2019a":
            first_col = 6
        else: 
            first_col = 9
    df = df[df.columns[first_col:]]  # removing "metadata" columns

    return df, df_raw

#adapt
def adapt_df(df, parties, include_no_vote=False, ballot_number_field_name=None):

    df['ballot_id'] = df['סמל ישוב'].astype(str) + '__' + df[ballot_number_field_name].astype(str)
    df_yeshuv = df.index  # new: keep yeshuv
    df = df.set_index('ballot_id')
    eligible_voters = df['בזב']
    total_voters = df['מצביעים']
    df = df[parties]
    df['ישוב'] = df_yeshuv  # new: keep yeshuv
    df = df.reindex(sorted(df.columns), axis=1)
    if include_no_vote:
        df['לא הצביע'] = eligible_voters - total_voters
    return df


def  subplot_bars(x, y, xticklabels="", title="", title_str="",texts=[],q=0, apr_lables=None,sep_lables = None):


    # Bar Plot votes for each eshkol in a different subplot. Use plt.subplots
    width = 0.25
    rev_names = [name[::-1] for name in list(xticklabels)]
    fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(16, 8))
    fig.suptitle(title, fontsize=16)

    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]
    n = len(axs)
    bar1 = []
    text=""
    tag=" "
    for i in range(len(axs)):
        bar1.append(axs[i].bar(np.arange(10), y.iloc[i,:], width, color='rebeccapurple'))
        bar1.append(axs[i].bar(np.arange(10)+width, x.iloc[i, :], width, color='deepskyblue'))
        if q == 2:
            tag = "Distance of rounds"
        if q==3:
            tag="Avg. voting %"
        if q==4:
            tag="Vote % change of rounds"
        if q==5:
            tag="Vote % change of rounds"
        text = "{0} in Ballot: {1}".format(tag, round(texts[i],2))


        leg= axs[i].legend(labels=["april {}/{}".format(apr_lables.iloc[i,0],apr_lables.iloc[i,1]),
                                   "september. {}/{}".format(sep_lables.iloc[i,0],sep_lables.iloc[i,1])],   # The labels for each line
               loc='upper right',   # Position of legend
               borderaxespad=0.05,    # Small spacing around legend box
                prop={'size':6},fontsize = 'x-small') # Title for the legend)
        leg.set_title("Month of 2019\nKohser/BZB",prop={'size':7})

        axs[i].set_ylabel('Votes Percentage')
        axs[i].set_title(title_str[i]+"\n"+text,fontsize=10)
        axs[i].set_xticks(np.arange(n))
        axs[i].set_xticklabels(xticklabels,rotation=90)
        axs[i].set_facecolor('whitesmoke')

        # set individual bar lables using above list
        for j in axs[i].patches:
            # get_x pulls left or right; get_height pushes up or down
            if round((j.get_height()) * 100, ndigits=0)>10:
                axs[i].text(j.get_x(), j.get_height() + 0.001,
                        str(round((j.get_height()) * 100, ndigits=0)), fontsize=6, color='grey',
                        rotation=0)

    # plt.style.use('seaborn')
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.set_facecolor('whitesmoke')
    plt.show()



#adapt
def adapt_df2(df, parties, include_no_vote=False, ballot_number_field_name=None):
    df['ballot_id'] = df['סמל ישוב'].astype(str) + '__' + df[ballot_number_field_name].astype(str)
    df_yeshuv = df.index  # new: keep yeshuv
    df = df.set_index('ballot_id')
    eligible_voters = df['בזב']
    total_voters = df['מצביעים']
    kosher = (df['כשרים'])
    #kosher = kosher.reindex(sorted(df.columns), axis=1)
    df = df[parties]
    df['ישוב'] = df_yeshuv  # new: keep yeshuv
    df = df.reindex(sorted(df.columns), axis=1)
    if include_no_vote:
        df['לא הצביע'] = eligible_voters - total_voters
    return df,kosher


