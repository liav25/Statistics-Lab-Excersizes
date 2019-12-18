import numpy as np

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

