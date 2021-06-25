import pandas as pd
import os, csv, re

nav_train_file = 'train.tsv'
nav_dev_file = 'test.tsv'
nav_file_path = r'.\data\naver-ner'

cus_train_file = 'train.tsv'
cus_dev_file = 'dev.tsv'
cus_file_path = r'.\data\custom-ner'

nav_train = pd.read_csv(os.path.join(nav_file_path, nav_train_file), header = None, sep="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')
nav_dev = pd.read_csv(os.path.join(nav_file_path, nav_dev_file), header = None, sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8')

cus_train = pd.read_csv(os.path.join(cus_file_path, cus_train_file), header = None, sep="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')
cus_dev = pd.read_csv(os.path.join(cus_file_path, cus_dev_file), header = None, sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8')

def prepro_tag(sentence, tag):
    tmp = []
    for w, t in zip(re.sub(' ', '_ ', sentence).split(), tag.split()):

        if '_' not in w :
            if t == 'O':
                tmp.extend([t]*len(w))
            else :
                if '.' in w :
                    tmp.extend([t[:2]+'-B'] + [t[:2]+'-I']*(len(w)-2) + ['O'])
                else :    
                    tmp.extend([t[:2]+'-B'] + [t[:2]+'-I']*(len(w)-1))
    
        else :
            if t == 'O' :
                tmp.extend([t]*len(w))
            elif t[-1] == 'B' :
                tmp.extend([t[:2]+'-B'] + [t[:2]+'-I']*(len(w)-2) + ['O'])
            elif t[-1] == 'I' :
                del tmp[-1]
                tmp.extend([t]*(len(w)) + ['O'])
        
    return " ".join(tmp)

def prepro_char(df):
    prepro_df = df.copy()

    if (df.apply(lambda x : len(x[0].split())!=len(x[1].split()), axis=1).sum()) != 0:
        return print("tags and words count mismatch")
    
    df[0] = df[0].apply(lambda x : x.strip())
    prepro_df['text'] = df[0].apply(lambda x : " ".join([char for char in re.sub(' ', '_', x)]))
    prepro_df['tag'] = df.apply(lambda x : prepro_tag(x[0], x[1]), axis=1)
    
    return prepro_df[['text', 'tag']]

p_nav_train = prepro_char(nav_train)
p_nav_dev = prepro_char(nav_dev)
p_cus_train = prepro_char(cus_train)
p_cus_dev = prepro_char(cus_dev)

p_nav_train.to_csv(r'.\data\p_nav_train.tsv', sep='\t', index=False, header=False)
p_nav_dev.to_csv(r'.\data\p_nav_dev.tsv', sep='\t', index=False, header=False)
p_cus_train.to_csv(r'.\data\p_cus_train.tsv', sep='\t', index=False, header=False)
p_cus_dev.to_csv(r'.\data\p_cus_dev.tsv', sep='\t', index=False, header=False)


