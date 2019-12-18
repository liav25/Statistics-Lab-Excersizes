# Functions for Loading and analysis of election data
# Modified from a python notebook from Harel Kein
# Call libraries
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import gridspec



pd.set_option('display.max_rows',10000000)



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