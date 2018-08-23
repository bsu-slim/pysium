'''
Created on Aug 22, 2018

@author: ckennington
'''
from sium import SIUM
import numpy as np
import pickle
from operator import itemgetter
import sqlite3
import pandas as pd
from scipy.spatial import distance
from pandasql import sqldf
from sklearn import ensemble
import random

sium = SIUM('take')
'''
requires take.db to be in the working folder
'''
print('preparing data...')
# connect to the database
con = sqlite3.connect('take.db')
# get features
tiles = pd.read_sql_query("SELECT * FROM piece", con)
# get referents
targs = pd.read_sql_query("SELECT * FROM referent", con)
# obtain the referring expressions as utts
utts = pd.read_sql_query("SELECT * FROM hand", con)

query = '''
SELECT tiles.* FROM
targs 
INNER JOIN
tiles
ON targs.episode_id = tiles.episode_id
AND targs.object = tiles.id;
'''
targets = sqldf(query, globals())

# the result of this shuold be the words and corresponding object features for positive examples
query = '''
SELECT utts.word, utts.inc, targets.* FROM
targets 
INNER JOIN
utts
ON targets.episode_id = utts.episode_id
'''
positive = sqldf(query, locals())

num_eval = 100
eids = set(positive.episode_id)
test_eids = set(random.sample(eids, num_eval))
train_eids = eids - test_eids
positive_train = positive[positive.episode_id.isin(train_eids)]
words = list(set(utts.word))

print("training...")
pword = '<pword>'
for i,row in positive_train.iterrows():
    word = row['word']
    sium.add_word_to_property(row['color'], {'word':word,'pword':pword})
    sium.add_word_to_property(row['type'], {'word':word,'pword':pword})
    sium.add_word_to_property(row['grid'], {'word':word,'pword':pword})
    pword = word

sium.train()
print('persisting and loading model...')
sium.persist_model()
sium.load_model()


print('performing evaluation...')
utts_eval = utts[utts.episode_id.isin(test_eids)]
tiles_eval = tiles[tiles.episode_id.isin(test_eids)]

query = '''
SELECT utts_eval.word, utts_eval.inc, tiles_eval.* FROM
utts_eval
INNER JOIN
tiles_eval
ON utts_eval.episode_id = tiles_eval.episode_id
'''
eval_data = sqldf(query, locals())

corr = []
for eid in list(set(eval_data.episode_id)):
    sium.new_utt()
    episode = eval_data[eval_data.episode_id == eid]
    pword = '<pword>'
    for inc in list(set(episode.inc)):
        increment = episode[episode.inc == inc]
        context = dict()
        for i,row in increment.iterrows():
            context[row['id']] = {'color':row['color'],'type':row['type'], 'grid':row['grid']}
            sium.set_context(context)
        word = increment.word.iloc[0] # all the words in the increment are the same, so just get the first one
        intents = increment.id
        p = sium.add_word_increment({'word':word,'pword':pword})
        pword = word
    corr.append(sium.get_predicted_intent()[0]==list(set(targs[targs.episode_id == eid].object))[0])
print('accuracy on test set of 100 random items:', sum(corr)/len(corr))
