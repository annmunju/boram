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

import pymysql

# story와 review 데이터 형태소 분리 작업

def wordSep(df):
    story = df['story']
    story = story.fillna('')

    review = df['review']
    review = review.fillna('')

    stopwords = {'하게', '그것', '되', '선보이는', '해서', '빠져', '있으며', '문학상', '소설가', '작품', '하는데', '온',
                '있었다', '로부터', '이제', '가는', '뒤', '하기', '받아', '되지', '도', '있을', '된', '과', '우리', '따라', 
                '하며', '했던', '앞', '받는', '되면서', '통해', '부', '해', '본', '일까', '그러던', '못', '있고', '또', 
                '하면', '없는', '없었던', '발간', '한다', '번', '있게', '해야', '나가는', '이어', '잘', '문장', '이자', 
                '국문학', '있을까', '더욱', '대한', '되는', '미', '관', '준다', '된다', '듯', '점', '중', '권', '네', '와', 
                '낸', '이런', '무엇', '하는', '냈다', '연재', '다섯', '입니다', '받으며', '로', '동안', '갈', '두', '세', 
                '현대문학', '수', '그려', '찾아', '출', '되는데', '모든', '있도록', '했다', '않은', '하면서', '없을', '있지만', 
                '내', '최고', '사는', '이번', '가지', '되고', '한', '속', '개정판', '다른', '될', '가장', '이후', '할', 
                '베스트셀러', '은', '는', '이었던', '선정', '이', '없었다', '알', '어떻게', '담아', '담겨', '안', '단편소설',
                '년', '없고', '담고', '후', '하던', '수상작', '또한', '그', '간', '있는', '만나', '않는다', '게', '위해', 
                '인물', '채', '있다고', '문학', '만들어', '많은', '대해', '감', '이래', '이러한', '싶은', '써', '번역', 
                '전', '시리즈', '않는', '호', '보다', '것', '같은', '독자', '담긴', '전집', '곳', '넘나', '때문', '있다면', 
                '등', '거', '이야기', '그렇다면', '담은', '작가', '소설', '있어', '해온', '넘어', '모두', '이름', '의해', 
                '첫', '같다', '않고', '저', '다룬', '편', '자', '이를', '주는', '장편소설', '지', '드는', '있', '였다', 
                '하나', '하여', '아닐', '김', '되었다', '제', '지금', '데', '데뷔', '있다', '작', '있었던', '아니라', 
                '한편', '위', '둘', '바로', '있다는', '하지', '낸다', '그런', '란', '였던', '다시', '받고', '를', '쓴', 
                '어떤', '하다', '이었다', '되어', '인해', '보여준다', '책', '저자', '신작', '더', '장', '대', '가진', '개', 
                '아니다', '가', '수록', '아닌', '하고', '없다', '않았다', '단편', '로서', '있던', '때', '문제', '의', 
                '하는', '된', '한', '한다', '할', '출', '해', '된다', '는', '하며', '했다', '되는', '하게', '되었다', 
                '되어', '가', '했던', '될', '하여', '그려', '낸', '되고', '하지', '하기', '와', '않는', '하면서', 
                '하다', '잘', '하고', '하는데', '쓴', '않고', '하던', '되는데', '않은', '담은', 
                '사는', '가진', '냈다', '대', '받고', '이었다', '해야', '가는', '해온', 
                '싶은', '받는', '이를', '않는다', '만나', '담아', '되면서', '담고', '다룬', '보여준다', '따라', 
                '보다',  '되지', '주는', '본', '찾아', '써', '넘어', 
                '넘나', '되', '선보이는', '준다', '해서', '일까', '만들어', '이었던', '받아', '받으며', 
                '담긴', '담겨', '갈', '하면', '나가는', '빠져', '낸다', '였다', '였던', '지', '이어', '드는' , '않았다',
                '발간', '문학', '권', '를', '있었다', '가장', '권', '이자', '두', '온', '제',
                '소설가', '김','그런', '곳', '더'}

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


conn=pymysql.connect(host='34.64.181.43', user='root', password='1234', db='novelmusic')
sql="select * from novel"
df = pd.read_sql_query(sql, conn)
df_lda = lda_make_df(df)
df_lda = df_lda[['id', 'Dominant_Topic', 'Topic_Keywords']]

cursor = conn.cursor()
for i,t,k in zip(df_lda['id'], df_lda['Dominant_Topic'], df_lda['Topic_Keywords']):
    sql_update = f'update novel set Dominant_Topic = {t}, Topic_Keywords = {k} where id = {i}'
    cursor.execute(sql_update)
    
conn.commit()

conn.close()



