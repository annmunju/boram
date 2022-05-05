from tqdm import tqdm
import operator

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
    for idx, cos in enumerate(cos_ls_20):
      story_to_song = story_to_song.append({'novel':novel_df['id'][n],'song':song_df['id'][idx]}, ignore_index=True)

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

    cos_ls_20 = [idx for idx, tensor in cos_dict[:10]]
    for idx, cos in enumerate(cos_ls_20):
      story_to_color = story_to_color.append({'novel':novel_df['id'][n],'color':color_df['id'][idx]}, ignore_index=True)

  return story_to_color