# Import libraries and set settings

import pandas as pd
from collections import defaultdict
import numpy as np
import sqlite3
import os.path

langs, w, types, families, feature_types = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
feat_names, feat_number = [], []

# Read file and convert it into a dictionary.

def build_df(file):
    df = pd.read_csv(file)
    df = df.to_dict('index')
    return df

# Create dictionaries and dataframes

feature_types['Phonology'] = ['1A', '2A', '3A', '4A', '5A', '6A', '7A', '8A', '9A', '10A', '10B', '11A', '12A', '13A',
                              '14A', '15A', '16A', '17A', '18A', '19A']

feature_types['Sign Languages'] = ['139A', '140A']

feature_types['Other'] = ['141A', '142A']

feature_types['Morphology'] = ['20A', '21A', '21B', '22A', '23A', '24A', '25A', '25B', '26A', '27A', '28A', '29A']

feature_types['Nominal Categories'] = ['30A', '31A', '32A', '33A', '34A', '35A', '36A', '37A', '38A', '39A', '39B',
                                       '40A', '41A', '42A', '43A', '44A', '45A', '46A', '47A', '48A', '49A', '50A',
                                       '51A', '52A', '53A', '54A', '55A', '56A', '57A']

feature_types['Nominal Syntax'] = ['58A', '58B', '59A', '60A', '61A', '62A', '63A', '64A']

feature_types['Verbal Categories'] = ['65A', '66A', '67A', '68A', '69A', '70A', '71A', '72A', '73A', '74A', '75A',
                                      '76A', '77A', '78A', '79A', '79B', '80A']

feature_types['Word Order'] = ['81A', '81B', '82A', '83A', '84A', '85A', '86A', '87A', '88A', '89A', '90A', '90B',
                               '90C', '90D', '90E', '90F', '90G', '91A', '92A', '93A', '94A', '95A', '96A', '97A',
                               '143A', '143B', '143C', '143D', '143E', '143F', '143G', '144A', '144B', '144C', '144D',
                               '144E', '144F', '144G', '144H', '144I', '144J', '144K', '144L', '144M', '144N', '144O',
                               '144P', '144Q', '144R', '144S', '144T', '144U', '144V', '144W', '144X', '144Y']

feature_types['Simple Clauses'] = ['98A', '99A', '100A', '101A', '102A', '103A', '104A', '105A', '106A', '107A', '108A',
                                   '108B', '109A', '109B', '110A', '111A', '112A', '113A', '114A', '115A', '116A',
                                   '117A', '118A', '119A', '120A', '121A']

feature_types['Complex Sentences'] = ['122A', '123A', '124A', '125A', '126A', '127A', '128A']

feature_types['Lexicon'] = ['129A', '130A', '130B', '131A', '132A', '133A', '134A', '135A', '136A', '136B', '137A',
                            '137B', '138A']

d = {}
list_df = ['language', 'codes', 'values', 'parameters']
elements = ['iso_code', 'glottocode', 'Name', 'latitude', 'longitude', 'genus', 'family', 'macroarea', 'countrycodes']

for el in list_df:
    d[el] = build_df(el + '.csv')

# Build language dictionary langs

for i in range(len(d['language'])):
    for j in elements:
        langs[d['language'][i]['wals_code']][j] = d['language'][i][j]

name = d['codes'][0]['Parameter_ID']  # 1A

# Build feature dictionary w

for i in range(len(d['codes'])):
    if d['codes'][i]['Parameter_ID'] == name:
        feat_names.append(d['codes'][i]['Name'])
        w[d['codes'][i]['Parameter_ID']]['n_values'] += 1

    else:
        w[d['codes'][i - 1]['Parameter_ID']]['Names'] = feat_names
        feat_names, feat_number = [], []
        feat_names.append(d['codes'][i]['Name'])
        w[d['codes'][i]['Parameter_ID']]['n_values'] += 1

    if i == len(d['codes']) - 1:
        w[d['codes'][i]['Parameter_ID']]['Names'] = feat_names

    name = d['codes'][i]['Parameter_ID']

# Feature values per language

for i in range(len(d['values'])):
    langs[d['values'][i]['Language_ID']][d['values'][i]['Parameter_ID']] = w[d['values'][i]['Parameter_ID']][
                                                                               'Names'].index(
        d['values'][i]['Value']) + 1

# Finish saving feature values per language.

for i in langs:
    langs[i]['%'] = (len(langs[i]) - len(elements)) * 100 / len(w)

for i in langs:
    for j in feature_types:
        count = 0
        for k in feature_types[j]:
            if langs[i][k] != 0:
                count += 1
        langs[i][j] = count / len(feature_types[j])

    families[langs[i]['family']][i] = langs[i]['%']

# Feature types

for i in feature_types:
    types[i]['n_features'] = len(feature_types[i])
    for j in feature_types[i]:
        types[i]['n_values'] += w[j]['n_values']
        w[j]['emb_dim'] = max(1, int(max(1, np.floor((w[j]['n_values'] + 1) / 10))))
        types[i]['total_dim'] += w[j]['emb_dim']

# Build dataframe

feature_names, list_languages = [], []

for i in feature_types:
    for j in feature_types[i]:
        feature_names.append(j)

for i in langs:
    list_languages.append(i)

# Build database

if not os.path.isfile('WalsValues.db'):

    db = sqlite3.connect('WalsValues.db')
    cursor = db.cursor()
    cursor.execute('''CREATE TABLE WalsValues("language" TEXT,"1A" INT, "2A" INT, "3A" INT, "4A" INT, "5A" INT, "6A" INT, "7A" INT, "8A" INT, "9A" INT, "10A" INT, "10B" INT, "11A" INT, "12A" INT, "13A" INT, "14A" INT, "15A" INT, "16A" INT, "17A" INT, "18A" INT, "19A" INT, "139A" INT, "140A" INT, "141A" INT, "142A" INT, "20A" INT, "21A" INT, "21B" INT, "22A" INT, "23A" INT, "24A" INT, "25A" INT, "25B" INT, "26A" INT, "27A" INT, "28A" INT, "29A" INT, "30A" INT, "31A" INT, "32A" INT, "33A" INT, "34A" INT, "35A" INT, "36A" INT, "37A" INT, "38A" INT, "39A" INT, "39B" INT, "40A" INT, "41A" INT, "42A" INT, "43A" INT, "44A" INT, "45A" INT, "46A" INT, "47A" INT, "48A" INT, "49A" INT, "50A" INT, "51A" INT, "52A" INT, "53A" INT, "54A" INT, "55A" INT, "56A" INT, "57A" INT, "58A" INT, "58B" INT, "59A" INT, "60A" INT, "61A" INT, "62A" INT, "63A" INT, "64A" INT, "65A" INT, "66A" INT, "67A" INT, "68A" INT, "69A" INT, "70A" INT, "71A" INT, "72A" INT, "73A" INT, "74A" INT, "75A" INT, "76A" INT, "77A" INT, "78A" INT, "79A" INT, "79B" INT, "80A" INT, "81A" INT, "81B" INT, "82A" INT, "83A" INT, "84A" INT, "85A" INT, "86A" INT, "87A" INT, "88A" INT, "89A" INT, "90A" INT, "90B" INT, "90C" INT, "90D" INT, "90E" INT, "90F" INT, "90G" INT, "91A" INT, "92A" INT, "93A" INT, "94A" INT, "95A" INT, "96A" INT, "97A" INT, "143A" INT, "143B" INT, "143C" INT, "143D" INT, "143E" INT, "143F" INT, "143G" INT, "144A" INT, "144B" INT, "144C" INT, "144D" INT, "144E" INT, "144F" INT, "144G" INT, "144H" INT, "144I" INT, "144J" INT, "144K" INT, "144L" INT, "144M" INT, "144N" INT, "144O" INT, "144P" INT, "144Q" INT, "144R" INT, "144S" INT, "144T" INT, "144U" INT, "144V" INT, "144W" INT, "144X" INT, "144Y" INT, "98A" INT, "99A" INT, "100A" INT, "101A" INT, "102A" INT, "103A" INT, "104A" INT, "105A" INT, "106A" INT, "107A" INT, "108A" INT, "108B" INT, "109A" INT, "109B" INT, "110A" INT, "111A" INT, "112A" INT, "113A" INT, "114A" INT, "115A" INT, "116A" INT, "117A" INT, "118A" INT, "119A" INT, "120A" INT, "121A" INT, "122A" INT, "123A" INT, "124A" INT, "125A" INT, "126A" INT, "127A" INT, "128A" INT, "129A" INT, "130A" INT, "130B" INT, "131A" INT, "132A" INT, "133A" INT, "134A" INT, "135A" INT, "136A" INT, "136B" INT, "137A" INT, "137B" INT, "138A" INT)''')
    db.commit()

    vals = []
    for i in list_languages:  # for each language
        l = []
        l.append(i)
        for f in feature_names:  # for each feature
            l.append(langs[i][f])
        vals.append(tuple(l))

    cursor.executemany(
    ''' INSERT INTO WalsValues(language, "1A", "2A", "3A", "4A", "5A", "6A", "7A", "8A", "9A", "10A", "10B", "11A", "12A", "13A", "14A", "15A", "16A", "17A", "18A", "19A", "139A", "140A", "141A", "142A", "20A", "21A", "21B", "22A", "23A", "24A", "25A", "25B", "26A", "27A", "28A", "29A", "30A", "31A", "32A", "33A", "34A", "35A", "36A", "37A", "38A", "39A", "39B", "40A", "41A", "42A", "43A", "44A", "45A", "46A", "47A", "48A", "49A", "50A", "51A", "52A", "53A", "54A", "55A", "56A", "57A", "58A", "58B", "59A", "60A", "61A", "62A", "63A", "64A", "65A", "66A", "67A", "68A", "69A", "70A", "71A", "72A", "73A", "74A", "75A", "76A", "77A", "78A", "79A", "79B", "80A", "81A", "81B", "82A", "83A", "84A", "85A", "86A", "87A", "88A", "89A", "90A", "90B", "90C", "90D", "90E", "90F", "90G", "91A", "92A", "93A", "94A", "95A", "96A", "97A", "143A", "143B", "143C", "143D", "143E", "143F", "143G", "144A", "144B", "144C", "144D", "144E", "144F", "144G", "144H", "144I", "144J", "144K", "144L", "144M", "144N", "144O", "144P", "144Q", "144R", "144S", "144T", "144U", "144V", "144W", "144X", "144Y", "98A", "99A", "100A", "101A", "102A", "103A", "104A", "105A", "106A", "107A", "108A", "108B", "109A", "109B", "110A", "111A", "112A", "113A", "114A", "115A", "116A", "117A", "118A", "119A", "120A", "121A", "122A", "123A", "124A", "125A", "126A", "127A", "128A", "129A", "130A", "130B", "131A", "132A", "133A", "134A", "135A", "136A", "136B", "137A", "137B", "138A") VALUES(?,?,?,?,?, ?, ?, ?,?, ?,?, ?,?, ?,?, ?,?, ?,?, ?,?, ?,?, ?,?, ?,?, ?,?, ?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?, ?, ?, ?, ?, ?, ?,?, ?, ?, ?,?,?, ?, ?, ?, ?, ?, ?,?, ?, ?, ?,?, ?, ?, ?, ?, ?, ?,?, ?, ?, ?, ?, ?, ?, ?,?, ?, ?, ?, ?, ?, ?,?, ?, ?, ?, ?, ?, ?, ?,?, ?,?, ?, ?, ?,?, ?,?, ?,?, ?,?, ?,?, ?,?, ?, ?, ?,?, ?,?, ?,?, ?, ?,?, ?,?, ?, ?, ?,?, ?, ?,?, ?,?, ?, ?, ?, ?, ?, ?,?, ?,?, ?, ?, ?, ?, ?,?, ?,?, ?,?, ?,?, ?,?, ?,?, ?, ?,?, ?,?, ?,?, ?,?,?, ?, ?, ?,?, ?,?, ?,?, ?,?, ?, ?, ?,?, ?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
    vals)
    db.commit()

if not os.path.isfile('FeaturesList.db'):

    db = sqlite3.connect('FeaturesList.db')
    cursor = db.cursor()
    cursor.execute('''CREATE TABLE FeaturesList("feature" TEXT, "n_values" INT, "emb_dim" INT)''')
    db.commit()

    vals = []
    for i in feature_names:    # for each feature
        l = []
        l.append(i)
        l.append(w[i]['n_values'])
        l.append(w[i]['emb_dim'])
        vals.append(tuple(l))

    cursor.executemany('''INSERT INTO FeaturesList(feature, "n_values", "emb_dim")VALUES(?,?,?)''',vals)
    db.commit()

if not os.path.isfile('FTInfos.db'):

    db = sqlite3.connect('FTInfos.db')
    cursor = db.cursor()
    cursor.execute('''CREATE TABLE FTInfos("featuretype" TEXT, "n_features" INT, "n_values" INT, "total_dim" INT)''')
    db.commit()

    vals = []
    for i in types:    # for each feature type
        l = []
        l.append(i)
        l.append(types[i]['n_features'])
        l.append(types[i]['n_values'])
        l.append(types[i]['total_dim'])
        vals.append(tuple(l))

    cursor.executemany('''INSERT INTO FTInfos(featuretype, "n_features", "n_values", "total_dim")VALUES(?,?,?,?)''',vals)
    db.commit()

if not os.path.isfile('FTList.db'):

    db = sqlite3.connect('FTList.db')
    cursor = db.cursor()
    cursor.execute('''CREATE TABLE FTList("featuretype" TEXT, "features" TEXT)''')
    db.commit()

    vals = []
    for i in feature_types:    # for each feature type
        l,f = [], []
        l.append(i)
        for el in feature_types[i]:
            f.append(el)
        f = ','.join(f)
        f=str(f)
        l.append(f)
        vals.append(tuple(l))

    cursor.executemany('''INSERT INTO FTList(featuretype, "features")VALUES(?,?)''',vals)
    db.commit()
