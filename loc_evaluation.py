
import pandas as pd
import difflib
import json

# ********************* levenshtein distance - from nltk with changes *******************

def _edit_dist_init(len1, len2):
    lev = []
    for i in range(len1):
        lev.append([0] * len2)  # initialize 2D array to zero
    for i in range(len1):
        lev[i][0] = i           # column 0: 0,1,2,3,4,...
    for j in range(len2):
        lev[0][j] = j           # row 0: 0,1,2,3,4,...
    return lev


def _edit_dist_step(lev, i, j, s1, s2, transpositions=False, cor_matrix=None):
    c1 = s1[i - 1]
    c2 = s2[j - 1]

    # skipping a character in s1
    a = lev[i - 1][j] + 1
    # skipping a character in s2
    b = lev[i][j - 1] + 1
    if cor_matrix is None:

        cor = 1
    else:
        cor = (1 - cor_matrix[c1, c2]) / 2  # to be between 0 and 1

    # substitution
    # c = lev[i - 1][j - 1] + (c1 != c2)
    c = lev[i - 1][j - 1] + cor * (c1 != c2)

    # transposition
    d = c + 1  # never picked by default
    if transpositions and i > 1 and j > 1:
        if s1[i - 2] == c2 and s2[j - 2] == c1:
            d = lev[i - 2][j - 2] + 1

    # pick the cheapest
    if cor_matrix is None:
        lev[i][j] = min(a, b, c, d)
    else:
        lev[i][j] = min(c, d)


def edit_distance(s1, s2, transpositions=False, cor_matrix=None):
    """
    This was modified to take correlations into account!

    Calculate the Levenshtein edit-distance between two strings.
    The edit distance is the number of characters that need to be
    substituted, inserted, or deleted, to transform s1 into s2.  For
    example, transforming "rain" to "shine" requires three steps,
    consisting of two substitutions and one insertion:
    "rain" -> "sain" -> "shin" -> "shine".  These operations could have
    been done in other orders, but at least three steps are needed.

    This also optionally allows transposition edits (e.g., "ab" -> "ba"),
    though this is disabled by default.

    :param s1, s2: The strings to be analysed
    :param transpositions: Whether to allow transposition edits
    :type s1: str
    :type s2: str
    :type transpositions: bool
    :rtype int
    """
    # set up a 2-D array
    len1 = len(s1)
    len2 = len(s2)
    lev = _edit_dist_init(len1 + 1, len2 + 1)

    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            _edit_dist_step(lev, i + 1, j + 1, s1, s2, transpositions=transpositions, cor_matrix=cor_matrix)
    return lev[len1][len2]


def gestalt_diff(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).ratio()

# ********************


def get_gold_xlsx(data_path="", converstion_dict=None,
                  cat_dict=None):
    """
    Make spacy docs from annotated documents in xlsx format (with multiple sheets)
    :param data_path:
    :return:
    """
    sheets = pd.read_excel(data_path + "test_set-derived_from_testimonies1_full - 13.4.xlsx", sheet_name=None, usecols="B:D")
    d = {}
    for t, df in sheets.items():
        df.dropna(inplace=True)
        locs = df['location'].str.rstrip().to_list()
        if cat_dict is not None:
            locs = [cat_dict[l] for l in locs]
        if converstion_dict is not None:
            locs = [converstion_dict[l] for l in locs]
        d[t] = [df['text'].to_list(), locs]
    return d


