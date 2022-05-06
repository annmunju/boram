import pandas as pd
import tqdm
import re

from konlpy.corpus import kolaw
from konlpy.tag import Okt, Hannanum, Kkma, Komoran, Twitter, Mecab
# import MeCab
from konlpy.utils import concordance, pprint

import matplotlib.pyplot as plt
import seaborn as sns

# 한글깨짐 설정
import matplotlib.font_manager as fm

# print(fm.findSystemFonts(fontpaths=None, fontext='ttf'))
# font_location = '/usr/share/fonts/truetype/NanumGothic.ttf'  #font 경로 설정
# font_name = fm.FontProperties(fname=font_location).get_name()
# plt.rc('font', family=font_name)
# plt.rc('font', size=12)        # 기본 폰트 크기

import warnings
warnings.filterwarnings('ignore')

import gensim
gensim.__version__
from gensim import corpora, models
from gensim.models import CoherenceModel
from gensim.test.utils import common_corpus
from gensim.models.wrappers import LdaMallet


# story와 review 데이터 형태소 분리 작업

def wordSep(df):
    story = df['story']
    story = story.fillna('')

    review = df['review']
    review = review.fillna('')

    stopwords = ('하는', '된', '한', '한다', '할', '출', '해', '된다', '는', '하며', '했다', '되는', '하게', '되었다', 
                '되어', '가', '했던', '될', '하여', '그려', '낸', '되고', '하지', '하기', '와', '않는', '하면서', 
                '하다', '잘', '하고', '하는데', '쓴', '않고', '하던', '되는데', '않은', '담은', 
                '사는', '가진', '냈다', '대', '받고', '이었다', '해야', '가는', '해온', 
                '싶은', '받는', '이를', '않는다', '만나', '담아', '되면서', '담고', '다룬', '보여준다', '따라', 
                '보다',  '되지', '주는', '본', '찾아', '써', '넘어', 
                '넘나', '되', '선보이는', '준다', '해서', '일까', '만들어', '이었던', '받아', '받으며', 
                '담긴', '담겨', '갈', '하면', '나가는', '빠져', '낸다', '였다', '였던', '지', '이어', '드는' , '않았다',
                '발간', '문학', '현대문학')

    necessary_type = ('Noun', 'Verb', 'Adjective')

    sep_story = []

    for s in story:
        okt = Okt()
        story_sep = okt.pos(s)
        sentence = ''
        for word,typ in story_sep:
            if (word not in stopwords) and (typ in necessary_type):
                sentence += word + ' '
        sep_story.append(sentence)

    sep_review = []

    for r in review:
        okt = Okt()
        review_sep = okt.pos(r)
        sentence = ''
        for word,typ in review_sep:
            if (word not in stopwords) and (typ in necessary_type):
                sentence += word + ' '
        sep_review.append(sentence)

    analysis_data = [s+r for s,r in zip(sep_story, sep_review)]

    return analysis_data

def lda_coh(stories, no_below=5, num_topics=4, data_word=None, data_words=None):
    
    # 이야기 전체 형태소 분리
    if data_word == None:
        data_words = []
        for story in stories:
            data = list(str(story).split())
            data_words.append(data)
        
    
    if data_word == 'stop':
        data_words = []
        for story in stories:
            data = list(str(story).split())
            data_words.append(data)
            
        return data_words
    
    # id2word, corpus 추출
    id2word = corpora.Dictionary(data_words)
    id2word.filter_extremes(no_below = no_below) # no_below 회 이하로 등장한 단어는 삭제
    texts = data_words
    corpus = [id2word.doc2bow(text) for text in texts]
    
    mallet_path = './mallet-2.0.8/bin/mallet' 
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, random_seed=12, id2word=id2word)

    coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word, coherence='c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    
    return texts, corpus, ldamallet, coherence_ldamallet

def lda_coh_id2(stories, no_below=5, num_topics=4, data_word=None, data_words=None):
    
    # 이야기 전체 형태소 분리
    if data_word == None:
        data_words = []
        for story in stories:
            data = list(str(story).split())
            data_words.append(data)
        
    
    if data_word == 'stop':
        data_words = []
        for story in stories:
            data = list(str(story).split())
            data_words.append(data)
            
        return data_words
    
    # id2word, corpus 추출
    id2word = corpora.Dictionary(data_words)
    id2word.filter_extremes(no_below = no_below) # no_below 회 이하로 등장한 단어는 삭제
    texts = data_words
    corpus = [id2word.doc2bow(text) for text in texts]
    
    mallet_path = './mallet-2.0.8/bin/mallet' 
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, random_seed=12, id2word=id2word)

    coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word, coherence='c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    
    return texts, corpus, ldamallet, coherence_ldamallet, id2word

def format_topics_sentences(texts, corpus, data_words, ldamallet, Data=df):
    # Init output

    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    #ldamallet[corpus]: lda_model에 corpus를 넣어 각 토픽 당 확률을 알 수 있음
    for i, row in enumerate(ldamallet[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamallet.show_topic(topic_num, topn=10)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    # print(type(sent_topics_df))

    # Add original text to the end of the output
    contents = pd.Series(data_words)
    # sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    sent_topics_df = pd.concat([Data, sent_topics_df], axis=1)

    return sent_topics_df

def lda_make_df(df):
    analysis_data = wordSep(df)
    texts, corpus, ldamallet, coherence_ldamallet, id2word = lda_coh_id2(analysis_data, num_topics=10)
    data_words = lda_coh(analysis_data, num_topics=10, data_word='stop')
    df_topic_sents_keywords = format_topics_sentences(texts, corpus, data_words, ldamallet, Data=df)

    return df_topic_sents_keywords