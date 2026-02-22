#!/usr/bin/env python
# coding: utf-8

# <h1> Amazon Apparel Recommendations </h1>
# 
# 

# In[2]:


#importing all the necessary packages.

from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import math
import time
import re
import os
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances
from matplotlib import gridspec
from scipy.sparse import hstack
import plotly
import plotly.figure_factory as ff
from plotly.graph_objs import Scatter, Layout

plotly.offline.init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")


# In[2]:


# loading the data using pandas' read_json file.
data = pd.read_json('tops_fashion.json')


# In[3]:


print ('Number of data points : ', data.shape[0], \
       'Number of features/variables:', data.shape[1])


# In[4]:


data.columns


# In[5]:


# Feature Selection
data = data[['asin', 'brand', 'color', 'medium_image_url', 'product_type_name', 'title', 'formatted_price']]


# In[6]:


# asian = amazon standard identification number


# In[7]:


print ('Number of data points : ', data.shape[0], \
       'Number of features:', data.shape[1])
data.head()


# In[8]:


data.describe()


# In[9]:


print(data['product_type_name'].describe())


# In[10]:


data['product_type_name'].unique()


# In[11]:


# finding the 10 most frequent product_type_names.
product_type_count = Counter(list(data['product_type_name']))
product_type_count.most_common(10)


# In[12]:


print(data['brand'].describe())


# In[13]:


# finding the 10 most frequent brand.
product_type_count = Counter(list(data['brand']))
product_type_count.most_common(10)


# In[14]:


print(data['color'].describe())


# In[15]:


# finding the 10 most frequent colors.
product_type_count = Counter(list(data['color']))
product_type_count.most_common(10)


# In[16]:


print(data['formatted_price'].describe())


# In[17]:


# finding the 10 most frequent formatted_price.
product_type_count = Counter(list(data['formatted_price']))
product_type_count.most_common(10)


# In[18]:


print(data['title'].describe())


# In[19]:


product_type_count = Counter(list(data['title']))
product_type_count.most_common(10)


# In[20]:


data.to_pickle('pickels/180k_apparel_data')


# In[21]:


data = data.loc[~data['formatted_price'].isnull()]
print('Number of data points After eliminating price=NULL :', data.shape[0])


# In[22]:


data =data.loc[~data['color'].isnull()]
print('Number of data points After eliminating color=NULL :', data.shape[0])


# #### We brought down the number of data points from 183K  to 28K.

# In[23]:


''' To download all the 28k imgs on the local disk
import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Create folder
os.makedirs("images/28k_images", exist_ok=True)

headers = {"User-Agent": "Mozilla/5.0"}

def download_image(row):
    try:
        url = row['medium_image_url']
        asin = row['asin']
        
        if pd.isna(url):
            return
        
        filepath = f"images/28k_images/{asin}.jpeg"
        
        # Skip if already downloaded
        if os.path.exists(filepath):
            return
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        img.convert("RGB").save(filepath)
        
    except:
        pass

# Run with 20 threads (can increase to 30–40 depending on internet)
with ThreadPoolExecutor(max_workers=20) as executor:
    list(tqdm(executor.map(download_image, [row for _, row in data.iterrows()]), total=len(data)))
'''


# In[24]:


data.to_pickle('pickels/28k_apparel_data')


# In[25]:


print(sum(data.duplicated('title')))


# In[26]:


data_sorted = data[data['title'].apply(lambda x: len(x.split())>4)]
print("After removal of products with short description:", data_sorted.shape[0])


# In[27]:


data_sorted.sort_values('title',inplace=True, ascending=False)
data_sorted.head()


# <pre>
# Titles 1:
# 16. woman's place is in the house and the senate shirts for Womens XXL White
# 17. woman's place is in the house and the senate shirts for Womens M Grey
# 
# Title 2:
# 25. tokidoki The Queen of Diamonds Women's Shirt X-Large
# 26. tokidoki The Queen of Diamonds Women's Shirt Small
# 27. tokidoki The Queen of Diamonds Women's Shirt Large
# 
# Title 3:
# 61. psychedelic colorful Howling Galaxy Wolf T-shirt/Colorful Rainbow Animal Print Head Shirt for woman Neon Wolf t-shirt
# 62. psychedelic colorful Howling Galaxy Wolf T-shirt/Colorful Rainbow Animal Print Head Shirt for woman Neon Wolf t-shirt
# 63. psychedelic colorful Howling Galaxy Wolf T-shirt/Colorful Rainbow Animal Print Head Shirt for woman Neon Wolf t-shirt
# 64. psychedelic colorful Howling Galaxy Wolf T-shirt/Colorful Rainbow Animal Print Head Shirt for woman Neon Wolf t-shirt
# </pre>

# In[29]:


indices = []
for i,row in data_sorted.iterrows():
    indices.append(i)


# In[30]:


import itertools
stage1_dedupe_asins = []
i = 0
j = 0
num_data_points = data_sorted.shape[0]
while i < num_data_points and j < num_data_points:
    
    previous_i = i

    # store the list of words of ith string in a, ex: a = ['tokidoki', 'The', 'Queen', 'of', 'Diamonds', 'Women's', 'Shirt', 'X-Large']
    a = data['title'].loc[indices[i]].split()

    # search for the similar products sequentially 
    j = i+1
    while j < num_data_points:

        # store the list of words of jth string in b, ex: b = ['tokidoki', 'The', 'Queen', 'of', 'Diamonds', 'Women's', 'Shirt', 'Small']
        b = data['title'].loc[indices[j]].split()

        # store the maximum length of two strings
        length = max(len(a), len(b))

        # count is used to store the number of words that are matched in both strings
        count  = 0

        # itertools.zip_longest(a,b): will map the corresponding words in both strings, it will appened None in case of unequal strings
        # example: a =['a', 'b', 'c', 'd']
        # b = ['a', 'b', 'd']
        # itertools.zip_longest(a,b): will give [('a','a'), ('b','b'), ('c','d'), ('d', None)]
        for k in itertools.zip_longest(a,b): 
            if (k[0] == k[1]):
                count += 1

        # if the number of words in which both strings differ are > 2 , we are considering it as those two apperals are different
        # if the number of words in which both strings differ are < 2 , we are considering it as those two apperals are same, hence we are ignoring them
        if (length - count) > 2: # number of words in which both sensences differ
            # if both strings are differ by more than 2 words we include the 1st string index
            stage1_dedupe_asins.append(data_sorted['asin'].loc[indices[i]])


            # start searching for similar apperals corresponds 2nd string
            i = j
            break
        else:
            j += 1
    if previous_i == i:
        break


# In[31]:


data = data.loc[data['asin'].isin(stage1_dedupe_asins)]


# In[32]:


print('Number of data points : ', data.shape[0])


# In[33]:


data.to_pickle('pickels/17k_apperal_data')


# In[35]:


data = pd.read_pickle('pickels/17k_apperal_data')


# In[36]:


import itertools

# Precompute tokenized titles
titles = data['title'].apply(lambda x: x.split())

visited = set()
stage2_dedupe_asins = []

for i in range(len(data)):
    
    if i in visited:
        continue
        
    visited.add(i)
    stage2_dedupe_asins.append(data['asin'].iloc[i])
    
    a = titles.iloc[i]
    set_a = set(a)
    
    for j in range(i+1, len(data)):
        
        if j in visited:
            continue
            
        b = titles.iloc[j]
        set_b = set(b)
        
        length = max(len(a), len(b))
        count = len(set_a.intersection(set_b))
        
        if (length - count) < 3:
            visited.add(j)


# In[37]:


data = data.loc[data['asin'].isin(stage2_dedupe_asins)]


# In[38]:


print('Number of data points after stage two of dedupe: ',data.shape[0])


# In[41]:


data.to_pickle('pickels/16k_apperal_data')


# #### Text Preprocessing

# In[3]:


data = pd.read_pickle('pickels/16k_apperal_data')


# In[4]:


# we use the list of stop words that are downloaded from nltk lib.
stop_words = set(stopwords.words('english'))
print ('list of stop words:', stop_words)

def nlp_preprocessing(total_text, index, column):
    if type(total_text) is not int:
        string = ""
        for words in total_text.split():
            # remove the special chars in review like '"#$@!%^&*()_+-~?>< etc.
            word = ("".join(e for e in words if e.isalnum()))
            # Conver all letters to lower-case
            word = word.lower()
            # stop-word removal
            if not word in stop_words:
                string += word + " "
        data[column][index] = string


# In[ ]:


import time
start_time = time.perf_counter()

# we take each title and we text-preprocess it.
for index, row in data.iterrows():
    nlp_preprocessing(row['title'], index, 'title')

end_time = time.perf_counter()
print("Time taken:", end_time - start_time)


# In[10]:


data.head()


# In[11]:


data.to_pickle('pickels/16k_apperal_data_preprocessed')


# #### Text based product similarity

# In[13]:


data = pd.read_pickle('pickels/16k_apperal_data_preprocessed')
data.head()


# In[14]:


# Utility Functions which we will use through the rest of the workshop.


#Display an image
def display_img(url,ax,fig):
    # we get the url of the apparel and download it
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    # we will display it in notebook 
    plt.imshow(img)
  
#plotting code to understand the algorithm's decision.
def plot_heatmap(keys, values, labels, url, text):
        # keys: list of words of recommended title
        # values: len(values) ==  len(keys), values(i) represents the occurence of the word keys(i)
        # labels: len(labels) == len(keys), the values of labels depends on the model we are using
                # if model == 'bag of words': labels(i) = values(i)
                # if model == 'tfidf weighted bag of words':labels(i) = tfidf(keys(i))
                # if model == 'idf weighted bag of words':labels(i) = idf(keys(i))
        # url : apparel's url

        # we will devide the whole figure into two parts
        gs = gridspec.GridSpec(2, 2, width_ratios=[4,1], height_ratios=[4,1]) 
        fig = plt.figure(figsize=(25,3))
        
        # 1st, ploting heat map that represents the count of commonly ocurred words in title2
        ax = plt.subplot(gs[0])
        # it displays a cell in white color if the word is intersection(lis of words of title1 and list of words of title2), in black if not
        ax = sns.heatmap(np.array([values]), annot=np.array([labels]))
        ax.set_xticklabels(keys) # set that axis labels as the words of title
        ax.set_title(text) # apparel title
        
        # 2nd, plotting image of the the apparel
        ax = plt.subplot(gs[1])
        # we don't want any grid lines for image and no labels on x-axis and y-axis
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # we call dispaly_img based with paramete url
        display_img(url, ax, fig)
        
        # displays combine figure ( heat map and image together)
        plt.show()
    
def plot_heatmap_image(doc_id, vec1, vec2, url, text, model):

    # doc_id : index of the title1
    # vec1 : input apparels's vector, it is of a dict type {word:count}
    # vec2 : recommended apparels's vector, it is of a dict type {word:count}
    # url : apparels image url
    # text: title of recomonded apparel (used to keep title of image)
    # model, it can be any of the models, 
        # 1. bag_of_words
        # 2. tfidf
        # 3. idf

    # we find the common words in both titles, because these only words contribute to the distance between two title vec's
    intersection = set(vec1.keys()) & set(vec2.keys()) 

    # we set the values of non intersecting words to zero, this is just to show the difference in heatmap
    for i in vec2:
        if i not in intersection:
            vec2[i]=0

    # for labeling heatmap, keys contains list of all words in title2
    keys = list(vec2.keys())
    #  if ith word in intersection(lis of words of title1 and list of words of title2): values(i)=count of that word in title2 else values(i)=0 
    values = [vec2[x] for x in vec2.keys()]
    
    # labels: len(labels) == len(keys), the values of labels depends on the model we are using
        # if model == 'bag of words': labels(i) = values(i)
        # if model == 'tfidf weighted bag of words':labels(i) = tfidf(keys(i))
        # if model == 'idf weighted bag of words':labels(i) = idf(keys(i))

    if model == 'bag_of_words':
        labels = values
    elif model == 'tfidf':
        labels = []
        for x in vec2.keys():
            # tfidf_title_vectorizer.vocabulary_ it contains all the words in the corpus
            # tfidf_title_features[doc_id, index_of_word_in_corpus] will give the tfidf value of word in given document (doc_id)
            if x in  tfidf_title_vectorizer.vocabulary_:
                labels.append(tfidf_title_features[doc_id, tfidf_title_vectorizer.vocabulary_[x]])
            else:
                labels.append(0)
    elif model == 'idf':
        labels = []
        for x in vec2.keys():
            # idf_title_vectorizer.vocabulary_ it contains all the words in the corpus
            # idf_title_features[doc_id, index_of_word_in_corpus] will give the idf value of word in given document (doc_id)
            if x in  idf_title_vectorizer.vocabulary_:
                labels.append(idf_title_features[doc_id, idf_title_vectorizer.vocabulary_[x]])
            else:
                labels.append(0)

    plot_heatmap(keys, values, labels, url, text)


# this function gets a list of wrods along with the frequency of each 
# word given "text"
def text_to_vector(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    # words stores list of all words in given string, you can try 'words = text.split()' this will also gives same result
    return Counter(words) # Counter counts the occurence of each word in list, it returns dict type object {word1:count}



def get_result(doc_id, content_a, content_b, url, model):
    text1 = content_a
    text2 = content_b
    
    # vector1 = dict{word11:#count, word12:#count, etc.}
    vector1 = text_to_vector(text1)

    # vector1 = dict{word21:#count, word22:#count, etc.}
    vector2 = text_to_vector(text2)

    plot_heatmap_image(doc_id, vector1, vector2, url, text2, model)


# In[15]:


from sklearn.feature_extraction.text import CountVectorizer
title_vectorizer = CountVectorizer()
title_features   = title_vectorizer.fit_transform(data['title'])
title_features.get_shape()


# In[23]:


print(title_features[0])


# In[27]:


def bag_of_words_model(doc_id, num_results):
    # doc_id: apparel's id in given corpus
    
    # pairwise_dist will store the distance from given input apparel to all remaining apparels
    # the metric we used here is cosine, the coside distance is mesured as K(X, Y) = <X, Y> / (||X||*||Y||)
    # http://scikit-learn.org/stable/modules/metrics.html#cosine-similarity
    pairwise_dist = pairwise_distances(title_features,title_features[doc_id])
    
    # np.argsort will return indices of the smallest distances
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    #pdists will store the smallest distances
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]

    #data frame indices of the 9 smallest distace's
    df_indices = list(data.index[indices])
    
    for i in range(0,len(indices)):
        # we will pass 1. doc_id, 2. title1, 3. title2, url, model
        get_result(indices[i],data['title'].loc[df_indices[0]], data['title'].loc[df_indices[i]], data['medium_image_url'].loc[df_indices[i]], 'bag_of_words')
        print('ASIN :',data['asin'].loc[df_indices[i]])
        print ('Brand:', data['brand'].loc[df_indices[i]])
        print ('Title:', data['title'].loc[df_indices[i]])
        print ('Euclidean similarity with the query image :', pdists[i])
        print('='*60)

#call the bag-of-words model for a product to get similar products.
bag_of_words_model(12566, 20) # change the index if you want to.


# #### TF-IDF based similarity

# In[29]:


tfidf_title_vectorizer = TfidfVectorizer(min_df = 0.0)
tfidf_title_features = tfidf_title_vectorizer.fit_transform(data['title'])


# In[30]:


def tfidf_model(doc_id, num_results):
    # doc_id: apparel's id in given corpus
    
    # pairwise_dist will store the distance from given input apparel to all remaining apparels
    # the metric we used here is cosine, the coside distance is mesured as K(X, Y) = <X, Y> / (||X||*||Y||)
    # http://scikit-learn.org/stable/modules/metrics.html#cosine-similarity
    pairwise_dist = pairwise_distances(tfidf_title_features,tfidf_title_features[doc_id])

    # np.argsort will return indices of 9 smallest distances
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    #pdists will store the 9 smallest distances
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]

    #data frame indices of the 9 smallest distace's
    df_indices = list(data.index[indices])

    for i in range(0,len(indices)):
        # we will pass 1. doc_id, 2. title1, 3. title2, url, model
        get_result(indices[i], data['title'].loc[df_indices[0]], data['title'].loc[df_indices[i]], data['medium_image_url'].loc[df_indices[i]], 'tfidf')
        print('ASIN :',data['asin'].loc[df_indices[i]])
        print('BRAND :',data['brand'].loc[df_indices[i]])
        print ('Eucliden distance from the given image :', pdists[i])
        print('='*125)
tfidf_model(12566, 20)
# in the output heat map each value represents the tfidf values of the label word, the color represents the intersection with inputs title


# In[46]:


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------------------------------------------------------
# Encode all titles once (Equivalent to building w2v_title)
# ------------------------------------------------------------------

titles = data['title'].tolist()

title_embeddings = model.encode(
    titles,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

# ------------------------------------------------------------------
# Recommendation Function (Equivalent to avg_w2v_model)
# ------------------------------------------------------------------

def sbert_model(doc_id, num_results=10):

    query_vec = title_embeddings[doc_id].reshape(1, -1)

    similarities = cosine_similarity(query_vec, title_embeddings).flatten()

    indices = similarities.argsort()[-num_results:][::-1]
    scores = similarities[indices]

    for i in range(len(indices)):
        idx = indices[i]

        print("TITLE :", data['title'].iloc[idx])
        print("ASIN :", data['asin'].iloc[idx])
        print("BRAND :", data['brand'].iloc[idx])
        print("Cosine Similarity :", scores[i])
        print("="*120)

# ------------------------------------------------------------------
# Heatmap Visualization (Sentence-Level)
# ------------------------------------------------------------------

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics.pairwise import cosine_similarity

# Load tokenizer + model (same base as SBERT)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
bert_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_token_embeddings(sentence):

    inputs = tokenizer(sentence, return_tensors="pt")
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    token_embeddings = outputs.last_hidden_state.squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(
        inputs['input_ids'].squeeze(0)
    )

    merged_tokens = []
    merged_embeddings = []

    current_token = ""
    current_embedding = None

    for token, embedding in zip(tokens, token_embeddings):

        if token in ['[CLS]', '[SEP]']:
            continue

        if token.startswith("##"):
            current_token += token[2:]
            current_embedding += embedding.numpy()
        else:
            if current_token != "":
                merged_tokens.append(current_token)
                merged_embeddings.append(current_embedding)

            current_token = token
            current_embedding = embedding.numpy()

    # append last token
    if current_token != "":
        merged_tokens.append(current_token)
        merged_embeddings.append(current_embedding)

    merged_embeddings = np.array(merged_embeddings)

    return merged_tokens, merged_embeddings



def heat_map_sbert(sentence1, sentence2, url):
    
    tokens1, emb1 = get_token_embeddings(sentence1)
    tokens2, emb2 = get_token_embeddings(sentence2)

    # Compute cosine similarity matrix between tokens
    sim_matrix = cosine_similarity(emb1, emb2)

    gs = gridspec.GridSpec(2, 2, width_ratios=[4,1], height_ratios=[2,1])
    fig = plt.figure(figsize=(15,15))

    ax = plt.subplot(gs[0])
    sns.heatmap(np.round(sim_matrix,3), annot=True,
                xticklabels=tokens2,
                yticklabels=tokens1)

    ax.set_title(sentence2)

    ax = plt.subplot(gs[1])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    display_img(url, ax, fig)

    plt.show()


# In[47]:


np.save("title_embeddings.npy", title_embeddings)


# In[48]:


title_embeddings = np.load("title_embeddings.npy")


# In[50]:


heat_map_sbert(
    data['title'].iloc[12566],
    data['title'].iloc[13000],
    data['medium_image_url'].iloc[13000]
)


# In[51]:


from sklearn.metrics.pairwise import cosine_similarity

def sbert_recommend(doc_id, num_results=10):

    # Get query embedding
    query_vec = title_embeddings[doc_id].reshape(1, -1)

    # Compute cosine similarity with all titles
    similarities = cosine_similarity(query_vec, title_embeddings).flatten()

    # Sort in descending order (higher cosine = more similar)
    indices = similarities.argsort()[::-1][1:num_results+1]  # skip self-match
    scores = similarities[indices]

    for i in range(len(indices)):

        idx = indices[i]

        # Show heatmap visualization
        heat_map_sbert(
            data['title'].iloc[doc_id],
            data['title'].iloc[idx],
            data['medium_image_url'].iloc[idx]
        )

        print("ASIN :", data['asin'].iloc[idx])
        print("BRAND :", data['brand'].iloc[idx])
        print("Cosine Similarity :", scores[i])
        print("="*120)


# In[55]:


sbert_recommend(12566, 20)


# #### Weighted similarity using brand and color.

# In[56]:


from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity

data['brand'].fillna("Not given", inplace=True)

brands = [x.replace(" ", "-") for x in data['brand'].values]
types = [x.replace(" ", "-") for x in data['product_type_name'].values]
colors = [x.replace(" ", "-") for x in data['color'].values]

brand_vectorizer = CountVectorizer()
brand_features = brand_vectorizer.fit_transform(brands)

type_vectorizer = CountVectorizer()
type_features = type_vectorizer.fit_transform(types)

color_vectorizer = CountVectorizer()
color_features = color_vectorizer.fit_transform(colors)

extra_features = hstack((brand_features, type_features, color_features)).tocsr()


# In[59]:


import plotly.figure_factory as ff
import plotly
import matplotlib.gridspec as gridspec

def sbert_brand_weighted_visual(doc_id, w_text, w_meta, num_results=10):

    # TEXT similarity
    query_vec = title_embeddings[doc_id].reshape(1, -1)
    text_sim = cosine_similarity(query_vec, title_embeddings).flatten()

    # METADATA similarity
    meta_sim = cosine_similarity(
        extra_features[doc_id],
        extra_features
    ).flatten()

    # Weighted fusion
    final_score = (w_text * text_sim + w_meta * meta_sim) / (w_text + w_meta)

    # Sort descending
    indices = final_score.argsort()[::-1][1:num_results+1]
    scores = final_score[indices]

    for i in range(len(indices)):

        idx = indices[i]

        # ----------- TABLE PART -----------
        data_matrix = [
            ['Asin','Brand','Color','Product type'],
            [data['asin'].iloc[doc_id],
             data['brand'].iloc[doc_id],
             data['color'].iloc[doc_id],
             data['product_type_name'].iloc[doc_id]],
            [data['asin'].iloc[idx],
             data['brand'].iloc[idx],
             data['color'].iloc[idx],
             data['product_type_name'].iloc[idx]]
        ]

        colorscale = [[0, '#1d004d'],[.5, '#f2e5ff'],[1, '#f2e5d1']]
        table = ff.create_table(data_matrix, index=True, colorscale=colorscale)
        plotly.offline.iplot(table)

        # ----------- HEATMAP + IMAGE -----------
        heat_map_sbert(
            data['title'].iloc[doc_id],
            data['title'].iloc[idx],
            data['medium_image_url'].iloc[idx]
        )

        print("Final Similarity Score :", scores[i])
        print("="*130)


# In[118]:


sbert_brand_weighted_visual(14605, 5, 5, 20)


# In[64]:


from scipy import sparse

sparse.save_npz("extra_features.npz", extra_features)


# In[65]:


import pickle

with open("brand_vectorizer.pkl", "wb") as f:
    pickle.dump(brand_vectorizer, f)

with open("type_vectorizer.pkl", "wb") as f:
    pickle.dump(type_vectorizer, f)

with open("color_vectorizer.pkl", "wb") as f:
    pickle.dump(color_vectorizer, f)


# In[66]:


import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))


# In[67]:


import tensorflow as tf

print("Default device:", tf.test.gpu_device_name())


# In[68]:


import tensorflow as tf
import time

# Large matrix multiplication (GPU-heavy task)
start = time.time()

with tf.device('/GPU:0'):
    a = tf.random.normal([8000, 8000])
    b = tf.random.normal([8000, 8000])
    c = tf.matmul(a, b)

print("Time taken:", time.time() - start)


# In[69]:


from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

print("Mixed precision enabled")


# In[76]:


import pandas as pd

data = pd.read_pickle("pickels/16k_apperal_data_preprocessed")

print(len(data))


# In[77]:


import os

image_dir = "images"
os.makedirs(image_dir, exist_ok=True)


# In[78]:


import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

headers = {
    "User-Agent": "Mozilla/5.0"
}

def download_image(row):
    asin = row['asin']
    url = row['medium_image_url']
    
    filepath = os.path.join(image_dir, f"{asin}.jpg")
    
    # Skip if already downloaded
    if os.path.exists(filepath):
        return
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
    except:
        pass  # silently ignore errors


rows = data.to_dict('records')

with ThreadPoolExecutor(max_workers=25) as executor:
    list(tqdm(executor.map(download_image, rows), total=len(rows)))


# In[79]:


print("Downloaded images:", len(os.listdir("images")))


# In[80]:


import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import os

# Load pretrained ResNet50 without classification head
model = ResNet50(
    weights='imagenet',
    include_top=False,
    pooling='avg'   # This gives 2048-d vector directly
)

print(model.output_shape)


# In[81]:


def extract_image_embedding(img_path):
    
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array, verbose=1)
    
    return features.flatten()   # 2048-d vector


# In[87]:


from tqdm import tqdm

batch_size = 4
image_paths = []
image_ids = []

for filename in os.listdir(image_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_paths.append(os.path.join(image_folder, filename))
        image_ids.append(filename)

image_embeddings = []

for i in tqdm(range(0, len(image_paths), batch_size)):
    
    batch_paths = image_paths[i:i+batch_size]
    batch_images = []
    
    for path in batch_paths:
        img = image.load_img(path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        batch_images.append(img_array)
    
    batch_images = np.array(batch_images)
    batch_images = preprocess_input(batch_images)
    
    features = model.predict(batch_images, verbose=1)
    image_embeddings.extend(features)

image_embeddings = np.array(image_embeddings)

np.save("image_embeddings.npy", image_embeddings)
np.save("image_ids.npy", np.array(image_ids))

print("Final shape:", image_embeddings.shape)


# In[88]:


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
image_embeddings = np.load("image_embeddings.npy")
image_ids = np.load("image_ids.npy")

# Load dataframe
data = pd.read_pickle("pickels/16k_apperal_data_preprocessed")

print(image_embeddings.shape)


# In[89]:


# Convert image_ids to set for faster lookup
available_asins = set([i.replace(".jpg","") for i in image_ids])

# Filter dataframe
image_data = data[data['asin'].isin(available_asins)].reset_index(drop=True)

print(len(image_data))


# In[92]:


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os


# In[93]:


def display_image(path, ax):
    img = Image.open(path)
    ax.imshow(img)
    ax.axis('off')


# In[94]:


def image_recommender_visual(query_index, top_k=5):
    
    query_vec = image_embeddings[query_index].reshape(1, -1)
    similarities = cosine_similarity(query_vec, image_embeddings)[0]
    similar_indices = similarities.argsort()[::-1][1:top_k+1]
    
    query_asin = image_data.iloc[query_index]['asin']
    query_path = os.path.join("images", query_asin + ".jpg")
    
    print("QUERY PRODUCT")
    print("ASIN:", query_asin)
    print("Title:", image_data.iloc[query_index]['title'])
    print("="*100)
    
    for idx in similar_indices:
        
        rec_asin = image_data.iloc[idx]['asin']
        rec_path = os.path.join("images", rec_asin + ".jpg")
        
        fig = plt.figure(figsize=(12,4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,2])
        
        # Query Image
        ax1 = plt.subplot(gs[0])
        display_image(query_path, ax1)
        ax1.set_title("Query Image")
        
        # Recommended Image
        ax2 = plt.subplot(gs[1])
        display_image(rec_path, ax2)
        ax2.set_title("Recommended")
        
        # Metadata display
        ax3 = plt.subplot(gs[2])
        ax3.axis('off')
        
        text = f"""
ASIN: {rec_asin}

Brand: {image_data.iloc[idx]['brand']}

Color: {image_data.iloc[idx]['color']}

Product Type: {image_data.iloc[idx]['product_type_name']}

Similarity Score: {similarities[idx]:.4f}
"""
        ax3.text(0.1, 0.5, text, fontsize=12, verticalalignment='center')
        
        plt.show()


# In[120]:


image_recommender_visual(1466 , top_k=10)


# In[121]:


import numpy as np
import pandas as pd
from scipy.sparse import load_npz

# Load data
data = pd.read_pickle("pickels/16k_apperal_data_preprocessed")

# Load embeddings
text_embeddings = np.load("title_embeddings.npy")
image_embeddings = np.load("image_embeddings.npy")
image_ids = np.load("image_ids.npy")

# Load metadata features
extra_features = load_npz("extra_features.npz")

print("Data shape:", len(data))
print("Text shape:", text_embeddings.shape)
print("Image shape:", image_embeddings.shape)
print("Meta shape:", extra_features.shape)


# In[122]:


# Remove .jpg from image_ids
available_asins = [asin.replace(".jpg","") for asin in image_ids]

# Create mapping from asin to original index
asin_to_index = {asin: idx for idx, asin in enumerate(data['asin'])}


# In[123]:


valid_indices = []

for asin in available_asins:
    if asin in asin_to_index:
        valid_indices.append(asin_to_index[asin])

print("Valid indices count:", len(valid_indices))


# In[124]:


# Filter dataframe
aligned_data = data.iloc[valid_indices].reset_index(drop=True)

# Filter text embeddings
aligned_text_embeddings = text_embeddings[valid_indices]

# Filter metadata
aligned_meta_features = extra_features[valid_indices]

# Image embeddings are already aligned to image_ids order
aligned_image_embeddings = image_embeddings

print("Aligned shapes:")
print("Data:", len(aligned_data))
print("Text:", aligned_text_embeddings.shape)
print("Image:", aligned_image_embeddings.shape)
print("Meta:", aligned_meta_features.shape)


# In[126]:


from sklearn.preprocessing import normalize

aligned_text_embeddings = normalize(aligned_text_embeddings)
aligned_image_embeddings = normalize(aligned_image_embeddings)
aligned_meta_features = normalize(aligned_meta_features)


# In[166]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def hybrid_recommender_mmr(query_index, top_k=5,
                           w_text=0.4, w_img=0.3, w_meta=0.3,
                           lambda_param=0.7,
                           category_boost=0.2,
                           price_boost=0.15,
                           price_tolerance=0.3):
    
    # -------------------------------
    # Compute Similarities
    # -------------------------------
    
    text_sim = cosine_similarity(
        aligned_text_embeddings[query_index].reshape(1, -1),
        aligned_text_embeddings
    )[0]
    
    img_sim = cosine_similarity(
        aligned_image_embeddings[query_index].reshape(1, -1),
        aligned_image_embeddings
    )[0]
    
    meta_sim = cosine_similarity(
        aligned_meta_features[query_index],
        aligned_meta_features
    )[0]
    
    # -------------------------------
    # Normalize Per Modality
    # -------------------------------
    
    text_sim = normalize_scores(text_sim)
    img_sim = normalize_scores(img_sim)
    meta_sim = normalize_scores(meta_sim)
    
    # -------------------------------
    # Hybrid Base Relevance
    # -------------------------------
    
    relevance = (
        w_text * text_sim +
        w_img * img_sim +
        w_meta * meta_sim
    )
    
    # Remove self
    relevance[query_index] = -1
    
    # -------------------------------
    # Soft Business Boosting
    # -------------------------------
    
    # Category boost
    query_category = aligned_data.iloc[query_index]['product_type_name']
    category_mask = (
        aligned_data['product_type_name'] == query_category
    ).values.astype(int)
    
    # Price preprocessing if needed
    if 'price_clean' not in aligned_data.columns:
        aligned_data['price_clean'] = (
            aligned_data['formatted_price']
            .str.replace('$', '', regex=False)
            .astype(float)
        )
    
    query_price = aligned_data.iloc[query_index]['price_clean']
    
    lower = query_price * (1 - price_tolerance)
    upper = query_price * (1 + price_tolerance)
    
    price_mask = (
        (aligned_data['price_clean'] >= lower) &
        (aligned_data['price_clean'] <= upper)
    ).values.astype(int)
    
    # Apply soft boosting instead of filtering
    relevance = relevance * (
        1
        + category_boost * category_mask
        + price_boost * price_mask
    )
    
    # -------------------------------
    # MMR Diversity Selection
    # -------------------------------
    
    selected = []
    candidates = list(range(len(relevance)))
    
    for _ in range(top_k):
        
        if not selected:
            idx = np.argmax(relevance)
            
            if relevance[idx] <= 0:
                break
            
            selected.append(idx)
            continue
        
        mmr_scores = []
        
        for idx in candidates:
            
            if idx in selected:
                mmr_scores.append(-1)
                continue
            
            if relevance[idx] <= 0:
                mmr_scores.append(-1)
                continue
            
            # Diversity penalty (image similarity)
            diversity_penalty = max(
                cosine_similarity(
                    aligned_image_embeddings[idx].reshape(1, -1),
                    aligned_image_embeddings[selected]
                )[0]
            )
            
            mmr_score = (
                lambda_param * relevance[idx]
                - (1 - lambda_param) * diversity_penalty
            )
            
            mmr_scores.append(mmr_score)
        
        idx = np.argmax(mmr_scores)
        
        if mmr_scores[idx] <= 0:
            break
        
        selected.append(idx)
    
    # -------------------------------
    # Display Results
    # -------------------------------
    
    print("\nQUERY PRODUCT")
    print("ASIN:", aligned_data.iloc[query_index]['asin'])
    print("Brand:", aligned_data.iloc[query_index]['brand'])
    print("Category:", aligned_data.iloc[query_index]['product_type_name'])
    print("Price:", aligned_data.iloc[query_index]['price_clean'])
    print("="*120)
    
    for idx in selected:
        print("ASIN:", aligned_data.iloc[idx]['asin'])
        print("Brand:", aligned_data.iloc[idx]['brand'])
        print("Category:", aligned_data.iloc[idx]['product_type_name'])
        print("Price:", aligned_data.iloc[idx]['price_clean'])
        print("Final Score:", round(relevance[idx], 4))
        print("="*120)
    
    for idx in top_indices:
        
        rec_asin = aligned_data.iloc[idx]['asin']
        rec_path = os.path.join("images", rec_asin + ".jpg")
        
        fig = plt.figure(figsize=(14,4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,2])
        
        # Query Image
        ax1 = plt.subplot(gs[0])
        display_image(query_path, ax1)
        ax1.set_title("Query")
        
        # Recommended Image
        ax2 = plt.subplot(gs[1])
        display_image(rec_path, ax2)
        ax2.set_title("Recommended")
        
        # Metadata Panel
        ax3 = plt.subplot(gs[2])
        ax3.axis('off')
        
        info = f"""
ASIN: {rec_asin}

Brand: {aligned_data.iloc[idx]['brand']}

Color: {aligned_data.iloc[idx]['color']}

Product Type: {aligned_data.iloc[idx]['product_type_name']}

Hybrid Score: {final_score[idx]:.4f}
(Text: {text_sim[idx]:.4f} | Meta: {meta_sim[idx]:.4f} | Image: {img_sim[idx]:.4f})
"""
        
        ax3.text(0.05, 0.5, info, fontsize=11, verticalalignment='center')
        
        plt.show()


# 
# # Hybrid Multi-Modal Recommender with Soft Business Boosting and Diversity Layer
# 
# ## Overview
# 
# This function implements a production-style hybrid recommendation system that integrates:
# 
# * Text similarity (Sentence-BERT embeddings)
# * Image similarity (ResNet embeddings)
# * Metadata similarity (Brand, Color, Product Type)
# * Soft business-aware boosting (Category and Price proximity)
# * Diversity-aware re-ranking using MMR (Maximal Marginal Relevance)
# 
# The system ensures recommendations are:
# 
# * Semantically relevant
# * Visually coherent
# * Commercially aligned
# * Non-redundant
# 
# ---
# 
# # Multi-Modal Similarity Computation
# 
# For a given `query_index`, cosine similarity is computed in three embedding spaces:
# 
# ```
# text_sim = cosine_similarity(...)
# img_sim = cosine_similarity(...)
# meta_sim = cosine_similarity(...)
# ```
# 
# ## Why cosine similarity?
# 
# Cosine similarity measures angular closeness between two vectors. It is well-suited for embedding spaces where direction matters more than magnitude.
# 
# Each modality captures different signals:
# 
# | Modality | Captures                                      |
# | -------- | --------------------------------------------- |
# | Text     | Semantic meaning of product title             |
# | Image    | Visual appearance (color, texture, pattern)   |
# | Metadata | Structured attributes like brand and category |
# 
# ---
# 
# # Per-Modality Normalization
# 
# Before fusion, each similarity vector is normalized using Min-Max scaling:
# 
# ```
# text_sim = normalize_scores(text_sim)
# img_sim = normalize_scores(img_sim)
# meta_sim = normalize_scores(meta_sim)
# ```
# 
# ## Why normalize?
# 
# Although cosine similarity ranges between 0 and 1, each modality may have different score distributions.
# Normalization ensures fair contribution during weighted fusion and prevents one modality from dominating due to distributional bias.
# 
# ---
# 
# # Hybrid Relevance Score (Late Fusion)
# 
# The base relevance score is computed as a weighted sum:
# 
# Relevance = w_text · S_text + w_img · S_img + w_meta · S_meta
# 
# ```
# relevance = (
#     w_text * text_sim +
#     w_img * img_sim +
#     w_meta * meta_sim
# )
# ```
# 
# ## Why Late Fusion?
# 
# Instead of combining raw embeddings (early fusion), we combine similarity scores. This approach provides:
# 
# * Modality-level control
# * Easier hyperparameter tuning
# * Interpretability
# * Independent embedding updates
# 
# ---
# 
# # Soft Business Boosting Layer
# 
# Instead of hard filtering, business rules are incorporated through score boosting.
# 
# ## Category Boost
# 
# Products belonging to the same category as the query receive a score multiplier:
# 
# ```
# relevance = relevance * (1 + category_boost * category_mask)
# ```
# 
# This encourages category consistency without eliminating cross-category exploration.
# 
# ---
# 
# ## Price Proximity Boost
# 
# Products within a defined tolerance range of the query price receive additional score boost:
# 
# ```
# lower = query_price * (1 - price_tolerance)
# upper = query_price * (1 + price_tolerance)
# ```
# 
# ```
# relevance = relevance * (1 + price_boost * price_mask)
# ```
# 
# This promotes economically relevant recommendations while preserving flexibility.
# 
# ---
# 
# ## Why Soft Boosting Instead of Hard Filtering?
# 
# Hard constraints:
# 
# * Remove candidates completely
# * Reduce exploration
# * May lead to empty result sets
# 
# Soft boosting:
# 
# * Adjusts ranking rather than eliminating items
# * Preserves diversity
# * Balances business logic with ranking flexibility
# 
# This approach more closely resembles production recommender systems.
# 
# ---
# 
# # Diversity Layer Using MMR (Maximal Marginal Relevance)
# 
# After computing the boosted relevance scores, results are re-ranked using MMR:
# 
# MMR = λ · Relevance − (1 − λ) · DiversityPenalty
# 
# Where:
# 
# * λ controls the relevance-diversity tradeoff
# * DiversityPenalty is the similarity between a candidate and already selected items
# 
# ```
# mmr_score = (
#     lambda_param * relevance[idx]
#     - (1 - lambda_param) * diversity_penalty
# )
# ```
# 
# ## Why MMR?
# 
# Without diversity control:
# 
# * Top recommendations may be near duplicates
# * Same brand or design may dominate
# * Results may appear repetitive
# 
# With MMR:
# 
# * First item maximizes relevance
# * Subsequent items balance relevance and diversity
# * Results feel curated rather than repetitive
# 
# ---
# 
# # Final Display Layer
# 
# The function:
# 
# * Prints query metadata
# * Prints selected recommendations
# * Displays side-by-side images
# * Shows final hybrid score
# 
# This improves transparency, interpretability, and presentation quality.
# 
# ---
# 
# # Architecture Summary
# 
# The recommender follows a three-stage pipeline:
# 
# ```
# 1. Multi-modal similarity computation
# 2. Soft business-aware score boosting
# 3. Diversity-aware re-ranking (MMR)
# ```
# 
# ---
# 
# # Why This Design Is Strong
# 
# Compared to a basic Top-K cosine similarity recommender:
# 
# | Basic Recommender  | This System                          |
# | ------------------ | ------------------------------------ |
# | Single modality    | Multi-modal fusion                   |
# | No business logic  | Business-aware boosting              |
# | Pure ranking       | Diversity-aware re-ranking           |
# | Repetitive results | Curated and balanced recommendations |
# 
# This architecture aligns with modern large-scale recommender system design principles.
# 
# ---
# 
# # Explanation (Short Version)
# 
# I built a multi-modal hybrid recommender that fuses text, image, and metadata similarities using weighted late fusion. Instead of applying rigid filters, I incorporated business logic through soft boosting for category and price proximity. Finally, I used Maximal Marginal Relevance to balance relevance and diversity, ensuring the recommendations are both commercially aligned and non-redundant.
# 
# ---
# 
# # Key Concepts Demonstrated
# 
# * Multi-modal embeddings
# * Cosine similarity in vector space
# * Score normalization
# * Late fusion architecture
# * Soft business-aware boosting
# * Diversity-aware ranking using MMR
# * Modular recommender system design
# 
# ---
# 
# If you would like, I can now also prepare:
# 
# * A formal system design section
# * A hyperparameter tuning explanation
# * A quantitative evaluation section
# * Or a resume-ready project summary suitable for data science roles

# In[168]:


hybrid_recommender_visual(1993)


# In[169]:


import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))


# In[172]:


import faiss
import numpy as np

text_vecs = aligned_text_embeddings.astype('float32')

dimension = text_vecs.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(text_vecs)

print("FAISS index built with", index.ntotal, "vectors")


# In[173]:


def faiss_retrieve(query_index, candidate_k=300):
    
    query_vector = text_vecs[query_index].reshape(1, -1)
    
    distances, indices = index.search(query_vector, candidate_k)
    
    return indices[0], distances[0]


# In[196]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def hybrid_recommender_faiss(query_index, top_k=20,
                             w_text=0.4, w_img=0.3, w_meta=0.3,
                             lambda_param=0.7,
                             category_boost=0.2,
                             price_boost=0.15,
                             price_tolerance=0.3,
                             candidate_k=300):

    # --------------------------------------------------
    # Stage 1: FAISS Retrieval (Text Embeddings)
    # --------------------------------------------------
    candidate_indices, text_scores = faiss_retrieve(query_index, candidate_k)

    # --------------------------------------------------
    # Stage 2: Compute Full Hybrid Similarities
    # --------------------------------------------------
    text_sim = text_scores.copy()

    img_sim = cosine_similarity(
        aligned_image_embeddings[query_index].reshape(1, -1),
        aligned_image_embeddings[candidate_indices]
    )[0]

    meta_sim = cosine_similarity(
        aligned_meta_features[query_index],
        aligned_meta_features[candidate_indices]
    )[0]

    # Normalize per modality (if not globally normalized)
    text_sim = normalize_scores(text_sim)
    img_sim = normalize_scores(img_sim)
    meta_sim = normalize_scores(meta_sim)

    # Hybrid weighted relevance
    relevance = (
        w_text * text_sim +
        w_img * img_sim +
        w_meta * meta_sim
    )

    # Remove self if present
    for i, idx in enumerate(candidate_indices):
        if idx == query_index:
            relevance[i] = -1

    # --------------------------------------------------
    # Stage 3: Soft Business Boosting
    # --------------------------------------------------

    # Category boost
    query_category = aligned_data.iloc[query_index]['product_type_name']

    category_mask = (
        aligned_data.iloc[candidate_indices]['product_type_name']
        == query_category
    ).values.astype(int)

    # Robust price cleaning
    if 'price_clean' not in aligned_data.columns:

        aligned_data['price_clean'] = (
            aligned_data['formatted_price']
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
        )

        aligned_data['price_clean'] = pd.to_numeric(
            aligned_data['price_clean'],
            errors='coerce'
        )

    query_price = aligned_data.iloc[query_index]['price_clean']

    if np.isnan(query_price):
        price_mask = np.zeros(len(candidate_indices))
    else:
        lower = query_price * (1 - price_tolerance)
        upper = query_price * (1 + price_tolerance)

        price_mask = (
            (aligned_data.iloc[candidate_indices]['price_clean'] >= lower) &
            (aligned_data.iloc[candidate_indices]['price_clean'] <= upper)
        ).values.astype(int)

    # Apply soft boosting
    relevance = relevance * (
        1 +
        category_boost * category_mask +
        price_boost * price_mask
    )

    # --------------------------------------------------
    # Stage 4: MMR Diversity Re-Ranking (Image Space)
    # --------------------------------------------------

    selected = []

    for _ in range(top_k):

        if not selected:
            idx = np.argmax(relevance)
            if relevance[idx] <= 0:
                break
            selected.append(idx)
            continue

        mmr_scores = []

        for i in range(len(candidate_indices)):

            if i in selected:
                mmr_scores.append(-1)
                continue

            if relevance[i] <= 0:
                mmr_scores.append(-1)
                continue

            diversity_penalty = max(
                cosine_similarity(
                    aligned_image_embeddings[candidate_indices[i]].reshape(1, -1),
                    aligned_image_embeddings[
                        candidate_indices[selected]
                    ]
                )[0]
            )

            mmr_score = (
                lambda_param * relevance[i]
                - (1 - lambda_param) * diversity_penalty
            )

            mmr_scores.append(mmr_score)

        idx = np.argmax(mmr_scores)

        if mmr_scores[idx] <= 0:
            break

        selected.append(idx)
    # --------------------------------------------------
    # Display Section (Your Preferred Format)
    # --------------------------------------------------

    query_asin = aligned_data.iloc[query_index]['asin']
    query_path = os.path.join("images", query_asin + ".jpg")

    print("\nQUERY PRODUCT")
    print("ASIN:", query_asin)
    print("=" * 120)

    for sel in selected:

        real_idx = candidate_indices[sel]

        rec_asin = aligned_data.iloc[real_idx]['asin']
        rec_path = os.path.join("images", rec_asin + ".jpg")

        fig = plt.figure(figsize=(14, 4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 2])

        # Query Image
        ax1 = plt.subplot(gs[0])
        display_image(query_path, ax1)
        ax1.set_title("Query")

        # Recommended Image
        ax2 = plt.subplot(gs[1])
        display_image(rec_path, ax2)
        ax2.set_title("Recommended")

        # Metadata Panel
        ax3 = plt.subplot(gs[2])
        ax3.axis('off')

        info = f"""
ASIN: {rec_asin}

Brand: {aligned_data.iloc[real_idx]['brand']}

Color: {aligned_data.iloc[real_idx]['color']}

Product Type: {aligned_data.iloc[real_idx]['product_type_name']}

Price: {aligned_data.iloc[real_idx]['price_clean']}

Final Hybrid Score: {relevance[sel]:.4f}

(Text: {text_sim[sel]:.4f} | Image: {img_sim[sel]:.4f} | Meta: {meta_sim[sel]:.4f})
"""

        ax3.text(0.05, 0.5, info, fontsize=11, verticalalignment='center')

        plt.show


# In[211]:


hybrid_recommender_faiss(1993)


# In[212]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def hybrid_recommender_faiss_select(query_index, top_k=20,
                             w_text=0.4, w_img=0.3, w_meta=0.3,
                             lambda_param=0.7,
                             category_boost=0.2,
                             price_boost=0.15,
                             price_tolerance=0.3,
                             candidate_k=300):

    # --------------------------------------------------
    # Stage 1: FAISS Retrieval (Text Embeddings)
    # --------------------------------------------------
    candidate_indices, text_scores = faiss_retrieve(query_index, candidate_k)

    # --------------------------------------------------
    # Stage 2: Compute Full Hybrid Similarities
    # --------------------------------------------------
    text_sim = text_scores.copy()

    img_sim = cosine_similarity(
        aligned_image_embeddings[query_index].reshape(1, -1),
        aligned_image_embeddings[candidate_indices]
    )[0]

    meta_sim = cosine_similarity(
        aligned_meta_features[query_index],
        aligned_meta_features[candidate_indices]
    )[0]

    # Normalize per modality (if not globally normalized)
    text_sim = normalize_scores(text_sim)
    img_sim = normalize_scores(img_sim)
    meta_sim = normalize_scores(meta_sim)

    # Hybrid weighted relevance
    relevance = (
        w_text * text_sim +
        w_img * img_sim +
        w_meta * meta_sim
    )

    # Remove self if present
    for i, idx in enumerate(candidate_indices):
        if idx == query_index:
            relevance[i] = -1

    # --------------------------------------------------
    # Stage 3: Soft Business Boosting
    # --------------------------------------------------

    # Category boost
    query_category = aligned_data.iloc[query_index]['product_type_name']

    category_mask = (
        aligned_data.iloc[candidate_indices]['product_type_name']
        == query_category
    ).values.astype(int)

    # Robust price cleaning
    if 'price_clean' not in aligned_data.columns:

        aligned_data['price_clean'] = (
            aligned_data['formatted_price']
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
        )

        aligned_data['price_clean'] = pd.to_numeric(
            aligned_data['price_clean'],
            errors='coerce'
        )

    query_price = aligned_data.iloc[query_index]['price_clean']

    if np.isnan(query_price):
        price_mask = np.zeros(len(candidate_indices))
    else:
        lower = query_price * (1 - price_tolerance)
        upper = query_price * (1 + price_tolerance)

        price_mask = (
            (aligned_data.iloc[candidate_indices]['price_clean'] >= lower) &
            (aligned_data.iloc[candidate_indices]['price_clean'] <= upper)
        ).values.astype(int)

    # Apply soft boosting
    relevance = relevance * (
        1 +
        category_boost * category_mask +
        price_boost * price_mask
    )

    # --------------------------------------------------
    # Stage 4: MMR Diversity Re-Ranking (Image Space)
    # --------------------------------------------------

    selected = []

    for _ in range(top_k):

        if not selected:
            idx = np.argmax(relevance)
            if relevance[idx] <= 0:
                break
            selected.append(idx)
            continue

        mmr_scores = []

        for i in range(len(candidate_indices)):

            if i in selected:
                mmr_scores.append(-1)
                continue

            if relevance[i] <= 0:
                mmr_scores.append(-1)
                continue

            diversity_penalty = max(
                cosine_similarity(
                    aligned_image_embeddings[candidate_indices[i]].reshape(1, -1),
                    aligned_image_embeddings[
                        candidate_indices[selected]
                    ]
                )[0]
            )

            mmr_score = (
                lambda_param * relevance[i]
                - (1 - lambda_param) * diversity_penalty
            )

            mmr_scores.append(mmr_score)

        idx = np.argmax(mmr_scores)

        if mmr_scores[idx] <= 0:
            break

        selected.append(idx)
    return [candidate_indices[sel] for sel in selected]


# In[213]:


hybrid_recommender_faiss_select(1993)


# In[214]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def evaluate_recommendations(query_index, selected_indices):
    
    print("\nEVALUATION METRICS")
    print("="*80)
    
    # Convert candidate-level indices to real dataset indices
    real_indices = selected_indices
    
    # ----------------------------
    # 1. Average Text Similarity
    # ----------------------------
    text_scores = cosine_similarity(
        aligned_text_embeddings[query_index].reshape(1,-1),
        aligned_text_embeddings[real_indices]
    )[0]
    
    avg_text_sim = np.mean(text_scores)
    print("Average Text Similarity:", round(avg_text_sim,4))
    
    # ----------------------------
    # 2. Average Image Similarity
    # ----------------------------
    img_scores = cosine_similarity(
        aligned_image_embeddings[query_index].reshape(1,-1),
        aligned_image_embeddings[real_indices]
    )[0]
    
    avg_img_sim = np.mean(img_scores)
    print("Average Image Similarity:", round(avg_img_sim,4))
    
    # ----------------------------
    # 3. Average Meta Similarity
    # ----------------------------
    meta_scores = cosine_similarity(
        aligned_meta_features[query_index],
        aligned_meta_features[real_indices]
    )[0]
    
    avg_meta_sim = np.mean(meta_scores)
    print("Average Meta Similarity:", round(avg_meta_sim,4))
    
    # ----------------------------
    # 4. Intra-List Diversity (Image Space)
    # ----------------------------
    rec_embeddings = aligned_image_embeddings[real_indices]
    
    pairwise_sim = cosine_similarity(rec_embeddings)
    
    # remove diagonal
    n = pairwise_sim.shape[0]
    diversity = 1 - (np.sum(pairwise_sim) - n) / (n*(n-1))
    
    print("Intra-List Diversity (Image):", round(diversity,4))
    
    # ----------------------------
    # 5. Brand Spread
    # ----------------------------
    brands = aligned_data.iloc[real_indices]['brand']
    unique_brands = len(set(brands))
    
    print("Unique Brands in Top-K:", unique_brands)
    
    # ----------------------------
    # 6. Price Deviation
    # ----------------------------
    query_price = aligned_data.iloc[query_index]['price_clean']
    rec_prices = aligned_data.iloc[real_indices]['price_clean']
    
    price_dev = np.mean(np.abs(rec_prices - query_price))
    print("Average Price Deviation:", round(price_dev,4))
    
    print("="*80)


# In[215]:


selected = hybrid_recommender_faiss_select(100)


# In[216]:


evaluate_recommendations(100, selected)


# In[217]:


import numpy as np
import matplotlib.pyplot as plt

def lambda_tradeoff_experiment(query_index,
                               lambdas=np.linspace(0.1, 0.95, 10),
                               top_k=20):
    
    relevance_scores = []
    diversity_scores = []
    
    for lam in lambdas:
        
        selected = hybrid_recommender_faiss_select(
            query_index=query_index,
            top_k=top_k,
            lambda_param=lam
        )
        
        # ---- Compute Relevance ----
        text_scores = cosine_similarity(
            aligned_text_embeddings[query_index].reshape(1,-1),
            aligned_text_embeddings[selected]
        )[0]
        
        img_scores = cosine_similarity(
            aligned_image_embeddings[query_index].reshape(1,-1),
            aligned_image_embeddings[selected]
        )[0]
        
        meta_scores = cosine_similarity(
            aligned_meta_features[query_index],
            aligned_meta_features[selected]
        )[0]
        
        avg_relevance = np.mean(
            0.4*text_scores + 0.3*img_scores + 0.3*meta_scores
        )
        
        relevance_scores.append(avg_relevance)
        
        # ---- Compute Diversity ----
        rec_embeddings = aligned_image_embeddings[selected]
        pairwise_sim = cosine_similarity(rec_embeddings)
        
        n = pairwise_sim.shape[0]
        diversity = 1 - (np.sum(pairwise_sim) - n) / (n*(n-1))
        
        diversity_scores.append(diversity)
    
    # -------------------------------
    # Plot Tradeoff
    # -------------------------------
    
    plt.figure(figsize=(10,5))
    
    plt.plot(lambdas, relevance_scores, marker='o', label="Average Relevance")
    plt.plot(lambdas, diversity_scores, marker='s', label="Intra-List Diversity")
    
    plt.xlabel("Lambda (Relevance vs Diversity)")
    plt.ylabel("Score")
    plt.title("Lambda Tradeoff Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return lambdas, relevance_scores, diversity_scores


# In[219]:


lambda_tradeoff_experiment(1992)


# ### Lambda controls the tradeoff between relevance and diversity in MMR. A higher lambda prioritizes similarity to the query, while a lower lambda encourages more diverse recommendations by penalizing similar items.

# #### 🔵 Blue Line → Average similarity of recommendations to the query  (Higher = more relevant)
# 
# #### 🟠 Orange Line → Diversity score (1 − average pairwise similarity among recommendations)  (Higher = more diverse)

# In[ ]:




