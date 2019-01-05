import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from collections import namedtuple
from gensim.utils import smart_open, to_utf8, tokenize
from nltk.corpus import stopwords
from itertools import *
import numpy as np
import re
import pandas as pd
import argparse
from progressbar import ProgressBar, Percentage, Bar, ETA
import warnings
warnings.filterwarnings("ignore")
def get_ngrams(text ):
    list1 =[]
    for i in range(1,3):
        n_grams = ngrams(word_tokenize(text), i)
        list1.append([ ' '.join(grams) for grams in n_grams])
    list1=list(chain.from_iterable(list1))
    return list1


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Smaartpulse")

    parser.add_argument('--file_name',
                        type=str,
                        help='')
    parser.add_argument('--text_column',
                        type=str)
    parser.add_argument('--output_file',
                        type=str)

    args = parser.parse_args()
    file_name = args.file_name
    text_column = args.text_column
    output_file = args.output_file
    output = pd.DataFrame()
    t =pd.read_csv(file_name)
    widgets = ["Generating POS tags: ", Percentage(), ' ', Bar(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=t.shape[0]).start()

    for i,row in enumerate(t.to_dict('responses')):
        response_1 = row[text_column]
        responses = re.split('[.;\n\r\t?!<>]',response_1)
        for response in responses:
            text=get_ngrams(response)
            if len(text)>0:
                pos=nltk.pos_tag(text)
                pos=pd.DataFrame(pos)
                pos.columns=['word','pos_tag']
                tree = pos[pos.pos_tag.str.contains('NN|VB') ]
            #     if tree.empty:
            #         tree = pos[pos.pos_tag.str.contains('VB') ]
                tree['word'] = tree['word'].astype('str')
                tree_map =tree['word'].str.len()>3
                tree_map=tree.loc[tree_map]
                listx = list(tree_map['word'])
                x = pos[pos.word.isin(listx)]
                x = x[~x.word.str.contains('/')]
                output = pd.concat([output,x])
        pbar.update(i)
    output=output.drop_duplicates()
    output =output[~output.word.str.contains('<|>|/|%|nobr')]
    pbar.finish()
    # top =output['word'].groupby(output['word']).value_counts()
    # final = pd.DataFrame([dict(count=values,name=key[0]) for key,values in top.iteritems()])    
    widgets = ["Generating Bi-Grams and Uni-Grams: ", Percentage(), ' ', Bar(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=output.shape[0]).start()

    output['temp_word'] = output['word'].apply(nltk.word_tokenize)
    output['word_pos'] = output['temp_word'].apply(nltk.pos_tag)
    item = np.array(['IN','PRP','DT','VBZ','``','=','TO','VBP','MD','PRP$','CC','POS'])
    aspect_uni =[]
    aspect_bi =[]

    for j,row in enumerate(output.to_dict('records')):
        flag=0
        np_array = np.array(zip(*row['word_pos'])[1])
        item_index =np.in1d(np_array,item).nonzero()[0]
        for j in range(0,len(item_index)):
            if j ==0 or j == len(np_array)-1:
                flag =flag+1
        if flag ==0:
            if len(row['word'].split())==1:
                aspect_uni.append(row['word'])
            else:
                aspect_bi.append(row['word'])
        if j != 0:
            pbar.update(j)
    aspect_uni = set(aspect_uni)
    aspect_bi = set(aspect_bi)
    final_out =[]
    for row in t.to_dict('records'):
        sent  = row[text_column]
        row['Unigarm__keywords'] = []
        row['Bigarm__keywords'] =[]
        st_ng =get_ngrams(sent)
        list1 = set(aspect_uni)&set(st_ng) 
        list2 = sorted(list1, key = lambda k : st_ng.index(k))
        if any(list2) :
            row['Unigarm__keywords'] = list2
        list1 = set(aspect_bi)&set(st_ng) 
        list2 = sorted(list1, key = lambda k : st_ng.index(k))
        if any(list2) :
            row['Bigarm__keywords'] = list2
        final_out.append(row)
    pbar.update(j+1)
    final_out = pd.DataFrame(final_out)
    final_out.to_csv(output_file,index=False)
    pbar.finish()
       