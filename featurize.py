import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import codecs
import re
import sys
import string

from nltk import word_tokenize, download, pos_tag, sent_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from datasketch import MinHash, MinHashLSH
from nltk.corpus import sentiwordnet as swn,SentiSynset
from mlxtend.preprocessing import TransactionEncoder

def infer_PCW(text_column):
  return text_column.apply(lambda row: 0 if len(row)==0 else sum(map(str.isupper, str(row).split()))/ len(str(row).split()) )

def infer_PC(text_column):
  return text_column.apply(lambda row: 0 if len(row)==0 else sum( 1 for c in row if c.isupper() ) / len( str(row) ) )

def infer_L(text_column):
  return text_column.str.split().str.len()

def infer_PP1(text_column):
  PP_1_Words = 'i','me','we','us','myself','ourselves','my', 'our','mine', 'ours'
  PP_2_Words = 'you', 'your', 'yours', 'yourself', 'yourselves'
  pronoun_tags = ['PRP','PRP$','WP','WP$'] 
  def calculate_pp1(line):
      words = word_tokenize(line)
      tagged_words = pos_tag(words)
      countPP_1 = 0
      countPP_2 = 0
      for tags in tagged_words:
          if tags[1] in pronoun_tags:
            if tags[0].lower() in PP_1_Words:
                countPP_1 += 1
            if tags[0].lower() in PP_2_Words:
                countPP_2 += 1
      countPro = countPP_1 + countPP_2
      if countPro > 0:
          percPP1 = (float(countPP_1)/countPro)
          #percPP2 = (float(countPP_2)/countPro)
      else:
          percPP1 = 0.0
          #percPP2 = 0.0
      return percPP1
    
  return text_column.apply(lambda row: 0 if len(row)==0 else calculate_pp1(row) )

def infer_RES(text_column):
  def calculate_res(line):
    tokenized_sentences = sent_tokenize(line)
    countExc = 0
    countSent = 0
    for sentence in tokenized_sentences:
        countSent += 1
        if '!' in sentence:
            countExc += 1
    if countSent > 0:
        ratioExcSent = float(countExc)/countSent
    else:
        ratioExcSent = 0.0
    return ratioExcSent

  return text_column.apply(lambda row: 0 if len(row)==0 else calculate_res(row) )


def infer_DLb(text_column, n_grams=2, is_checkpoint=False):

  text_copy = text_column.copy()

  if is_checkpoint:
    # this turns strings representing lists of tokens into actual lists, useful when using a checkpoint ds
    text_copy.loc[text_copy=='[]'] = "[' ',' ']"
    text_copy = text_copy.apply(lambda row: eval(row))

  transactions = text_copy.apply(lambda row: ngrams(row, n_grams) )
  print('finished calculating n-grams')

  transactions_copy = text_copy.apply(lambda row: ngrams(row, n_grams) ) # not proud of this but it works with fairly little memory
  print('finished calculating n-grams copy')
  

  te = TransactionEncoder()
  D = te.fit_transform(transactions)
  print('finished calculating code matrix')


  frecuencies = D.sum(axis=0)
  N= frecuencies.sum()
  probabilities = frecuencies/N
  entropies = []

    

  entropies = transactions_copy.apply(lambda row: sum( [-np.log2(probabilities[ngram]) for ngram in row]) )
  return entropies
  

def infer_DLu(text_column, is_checkpoint=False):

  text_copy = text_column.copy()

  if is_checkpoint:
    # this turns strings representing lists of tokens into actual lists, useful when using a checkpoint ds
    text_copy.loc[text_copy=='[]'] = "[' ']"
    text_copy = text_copy.apply(lambda row: eval(row))


  te = TransactionEncoder()
  D = te.fit(text_column)
  D= te.transform()

  print('finished calculating code matrix')
  
  frecuencies = D.sum(axis=0)
  N= frecuencies.sum()
  probabilities = frecuencies/N

  entropies = text_column.apply(sum( [-np.log2(probabilities[word]) for word in row]) )  


  return entropies

def infer_F(text_column, n_grams=3, Jaccard_threshold=0.5, num_perm=128 ):
  ''' Approximately match Strings using Locality sensitive hashing and calculates frecuency'''
  def calculate_LSH(data):
    
    data = text_column
    candidategroups = []

    # Create an MinHashLSH index optimized for Jaccard threshold 0.5,
    # that accepts MinHash objects with 128 permutations functions
    lsh = MinHashLSH(threshold=Jaccard_threshold, num_perm=num_perm)

    # Create MinHash objects
    minhashes = {}
    print('lhs initialized')

    for c, i in enumerate(data):

      
      minhash = MinHash(num_perm=128)
      for d in ngrams(i, n_grams):

        minhash.update("".join(d).encode('utf-8'))
      lsh.insert(c, minhash)
      minhashes[c] = minhash
    print('min hash generated')

    for i in range(len(minhashes.keys())):
      result = lsh.query(minhashes[i])
      candidategroups = candidategroups + [result]

    print('candidate groups generated')
    return candidategroups  

  result = text_column.to_frame()
  candidategroups = calculate_LSH(text_column)
  f = list(map(len, candidategroups))

  
  result['F'] = f

  # returns a dataframe
  return result
  
def infer_sw_ow(text_column):
  tag = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'TO', 'UH', 'PDT', 'SYM', 'RP']
  noun = ['NN', 'NNS', 'NP', 'NPS']
  adj = ['JJ', 'JJR', 'JJS']
  pronoun = ['PP', 'PP$', 'WP', 'WP$']
  verb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
  adverb = ['RB', 'RBR', 'RBS', 'WRB']
  regex = re.compile('[%s]' % re.escape(string.punctuation))
  
  def calculate_sw_ow(text):
      Tex = regex.sub(u'', text)
      words = word_tokenize(Tex.lower())
      word = pos_tag(words)
      objCount = 0
      subCount = 0
      for w in word:
          if not w[1] in tag:
              if w[1] in noun:
                  pos_Char = 'n'
              elif w[1] in adj:
                  pos_Char = 'a'
              elif w[1] in pronoun:
                  pos_Char = 'p'
              elif w[1] in verb:
                  pos_Char = 'v'
              elif w[1] in adverb:
                  pos_Char = 'r'
              else:
                  pos_Char = 'none'

              if pos_Char == 'none':
                  try:
                      
                      s = swn.senti_synsets(w[0])
                      scores = list(s)[0]
                      if scores.obj_score() > 0.5:
                          objCount += 1
                      elif scores.pos_score() + scores.neg_score() > 0.5:
                          subCount += 1
                  except:
                      pass #print('Unexpected word:', w[0])
              else:
                  try:
                      
                      s = swn.senti_synsets(w[0], pos_Char)
                      scores=list(s)[0]
                      if scores.obj_score() > 0.5:
                          objCount += 1
                      elif scores.pos_score() + scores.neg_score() > 0.5:
                          subCount += 1
                  except:
                      pass #print('Unexpected word:', w[0])

      if objCount+subCount > 0:
          ratioObj = float(objCount)/(objCount+subCount)
          ratioSub = float(subCount)/(objCount+subCount)
      else:
          ratioObj = 0.0
          ratioSub = 0.0

      return ratioObj, ratioSub

  series = text_column.apply(lambda row: (0, 0) if len(row)==0 else calculate_sw_ow(row) )


  columns = ("OW SW").split()
  df = pd.DataFrame([[ow, sw] for ow,sw, in series.values], columns=columns)

  # returns a dataframe
  return df

def text_featurize(df, text_column):

  df['PCW'] = infer_PCW(df[text_column])
  df['PC'] = infer_PC(df[text_column])
  df['L'] = infer_L(df[text_column])
  df['PP1'] = infer_PP1(df[text_column])
  df['RES'] = infer_RES(df[text_column])
  temp_df = infer_sw_ow(df[text_column])
  df['SW'] = temp_df['SW']
  df['OW'] = temp_df['OW']

  return df
