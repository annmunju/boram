from tqdm import tqdm
import operator
import pandas as pd
import numpy as np
import pymysql

def cos_similarity(v1, v2):
  dot_product = np.dot(v1, v2)
  norm = (np.sqrt(sum(np.square(v1))) * np.sqrt(sum(np.square(v2))))
  similarity = dot_product / norm

  return similarity

def novel_to_sing(novel_df, song_df):
  v1 = novel_df['vector']
  v2 = song_df['vector']

  story_to_song = pd.DataFrame(columns=['novel', 'song'])

  for n in tqdm(range(len(v1))):

    cos_sim_dict = {}
    for i in range(len(v2)):
      cos_sim_dict[i] = cos_similarity(v1[n], v2[i])
    cos_dict = sorted(cos_sim_dict.items(), key=operator.itemgetter(1), reverse=True)

    cos_ls_20 = [idx for idx, tensor in cos_dict[:20]]
    print(cos_ls_20)
    for idx, cos in enumerate(cos_ls_20):
      story_to_song = story_to_song.append({'novel':novel_df['id'][n],'song':song_df['id'][cos]}, ignore_index=True)

  return story_to_song

def novel_to_color(novel_df, color_df):
  v1 = novel_df['vector']
  v2 = color_df['vector']

  story_to_color = pd.DataFrame(columns=['novel', 'color'])

  for n in tqdm(range(len(v1))):

    cos_sim_dict = {}
    for i in range(len(v2)):
      cos_sim_dict[i] = cos_similarity(v1[n], v2[i])
    cos_dict = sorted(cos_sim_dict.items(), key=operator.itemgetter(1), reverse=True)

    cos_ls_10 = [idx for idx, tensor in cos_dict[:10]]
    for idx, cos in enumerate(cos_ls_10):
      story_to_color = story_to_color.append({'novel':novel_df['id'][n],'color':color_df['id'][cos]}, ignore_index=True)

  return story_to_color


# 데이터 입력
conn=pymysql.connect(host='34.64.181.43', user='root', password='1234', db='novelmusic')

sql_music="""select * from music"""
song = pd.read_sql_query(sql_music, conn)

sql_novel="""select * from novel fields terminated by '\t'"""
novel = pd.read_sql_query(sql_novel, conn)

sql_color="""select * from color"""
color = pd.read_sql_query(sql_color, conn)


novel_to_color_df = novel_to_color(novel, color)
novel_to_sing_df = novel_to_sing(novel, song)

cursor = conn.cursor()

sql_truncate = 'truncate novel_color'
cursor.execute(sql_truncate)
sql_truncate = 'truncate novel_sing'
cursor.execute(sql_truncate)

conn.commit()

for n,s in zip(novel_to_sing_df['novel'], novel_to_sing_df['song']):
    sql_insert = f'insert into novel_sing (novel, song) value ({n}, {s})'
    cursor.execute(sql_insert)
    
conn.commit()

for n,c in zip(novel_to_color_df['novel'], novel_to_color_df['color']):
    sql_insert = f'insert into novel_color (novel, color) value ({n}, {c})'
    cursor.execute(sql_insert)
    
conn.commit()

conn.close()