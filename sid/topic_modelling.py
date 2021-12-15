#%%
import os
import json
import gensim
from gensim.utils import simple_preprocess
import nltk
from nltk import corpus
# nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim.corpora as corpora
from pprint import pprint
import pyLDAvis.gensim_models
import pickle 
import pyLDAvis
import argparse
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import LdaModel, CoherenceModel
from gensim import corpora
from gensim.test.utils import datapath

DIR = "./data-with-sentiment"
COMMENTS_FILE_NAME = 'comments.json'
CAT_FILE_NAME = DIR+'/subsets.txt'
stop_words = stopwords.words('english')
cats = {}
with open(CAT_FILE_NAME) as file:
    for line in file:
        temp = line.split(': ')
        cats[temp[0]] = [id.strip('\'') for id in temp[1].strip('[').strip(']\n').split(', ')]
# print(cats)

def get_comments():
    if os.path.isfile(COMMENTS_FILE_NAME):
        with open(COMMENTS_FILE_NAME, 'r') as f:
            comments = json.load(f)
            return comments
    else:
        print('No summary available')
        exit(0)

def transform_data(comments):
    ids = []
    idToIndex = {}
    comments_list=[]
    ind=0
    for k,v in comments.items():
        ids.append(k)
        idToIndex[ind]=k
        ind+=1
        comments_list.append(' '.join(v))
    return ids, idToIndex, comments_list

def get_index_of_id(idToIndex, id):
    return idToIndex[id]

def get_id_of_index(ids, index):
    return ids[index]

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
            if word not in stop_words] for doc in texts]

def preprocess(comments):
    ids, idToIndex, comments_list = transform_data(comments)
    # comments_lol=[comment.split(" ") for comment in comments_list]
    data_words = list(sent_to_words(comments_list))
    # remove stop words
    data_words = remove_stopwords(data_words)
    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    return corpus, id2word, idToIndex, data_words

def save_csv(data, outfile):
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

def topic_model(num_topics):
    print('Starting topic modelling...')
    comments = get_comments()
    print('Loaded comments. Now, preprocessing...')
    for k, comments_cat in comments.items():
        if k=='YouTube - gaming':
            num_topics=10
        elif k=='YouTube - news':
            num_topics=5
        else:
            num_topics=14
        corpus, id2word, idToIndex, comments_lol = preprocess(comments_cat)
        # Build LDA model
        print('Loading model...')
        lda_model = gensim.models.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics)
        print('Loaded model.')
        # Print the Keyword in the 10 topics
        pprint(lda_model.print_topics())
        doc_lda = lda_model[corpus]
        print(lda_model.get_document_topics(corpus))
        post_topics = []
        topic_posts = []
        for i in range(num_topics):
            topic_posts.append([])
        for i, row_list in enumerate(doc_lda):
            print(i, row_list)
            th = 1.0/len(row_list)
            if len(row_list)==1:
                th=0
            topics=[]
            for topic in row_list:
                if topic[1] >= th:
                    topics.append(topic[0])
                    topic_posts[topic[0]].append(get_index_of_id(idToIndex, i))
            post_topics.append(topics)
            save_csv(post_topics, "post_topics"+k+".csv")
            save_csv(topic_posts, "topic_posts"+k+".csv")
        # return topic_posts, post_topics
        # Visualize the topics
        print('Visualizing...')
        # pyLDAvis.enable_notebook()
        LDAvis_data_filepath = 'ldavis/ldavis_prepared_'+k+str(num_topics)
        # # this is a bit time consuming - make the if statement True
        # # if you want to execute visualization prep yourself
        if True:
            LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
            with open(LDAvis_data_filepath, 'wb') as f:
                pickle.dump(LDAvis_prepared, f)
        # load the pre-prepared pyLDAvis data from disk
        with open(LDAvis_data_filepath, 'rb') as f:
            LDAvis_prepared = pickle.load(f)
        pyLDAvis.save_html(LDAvis_prepared, LDAvis_data_filepath +'.html')
        # return LDAvis_prepared, doc_lda, corpus

def jaccard_similarity(topic_1, topic_2):
    """
    Derives the Jaccard similarity of two topics

    Jaccard similarity:
    - A statistic used for comparing the similarity and diversity of sample sets
    - J(A,B) = (A ∩ B)/(A ∪ B)
    - Goal is low Jaccard scores for coverage of the diverse elements
    """
    intersection = set(topic_1).intersection(set(topic_2))
    union = set(topic_1).union(set(topic_2))
                    
    return float(len(intersection))/float(len(union))

def optimal_topics():
    comments = get_comments()
    for k, comments_cat in comments.items():
        print(k)
        print('Loaded comments. Now, preprocessing...')
        corpus, id2word, idToIndex, comments_lol = preprocess(comments_cat)
        # print('corpus')
        # print(corpus)
        # print('id2word')
        # print(id2word)
        # print(comments_lol)
        num_topics = list(range(16)[1:])
        num_keywords = 15

        LDA_models = {}
        LDA_topics = {}
        for i in num_topics:
            cwd = os.getcwd()
            temp_file = datapath(cwd+"/models/model"+k.split(' ')[-1]+str(i))
            # print(temp_file)
            if os.path.isfile(temp_file):
                print('Loading model file...')
                LDA_models[i] = LdaModel.load(temp_file)
            else:
                LDA_models[i] = gensim.models.LdaModel(corpus=corpus,
                                    id2word=id2word,
                                    num_topics=i)
                print('Saving model...')
                LDA_models[i].save(temp_file)

            shown_topics = LDA_models[i].show_topics(num_topics=i, 
                                                    num_words=num_keywords,
                                                    formatted=False)
            LDA_topics[i] = [[word[0] for word in topic[1]] for topic in shown_topics]
        LDA_stability = {}
        for i in range(0, len(num_topics)-1):
            jaccard_sims = []
            for t1, topic1 in enumerate(LDA_topics[num_topics[i]]): # pylint: disable=unused-variable
                sims = []
                for t2, topic2 in enumerate(LDA_topics[num_topics[i+1]]): # pylint: disable=unused-variable
                    sims.append(jaccard_similarity(topic1, topic2))    
                
                jaccard_sims.append(sims)
            
            LDA_stability[num_topics[i]] = jaccard_sims
        mean_stabilities = [np.array(LDA_stability[i]).mean() for i in num_topics[:-1]]
        # print(CoherenceModel(model=LDA_models[1], texts=comments_lol, dictionary=id2word, coherence='c_v').get_coherence())
        coherences = [CoherenceModel(model=LDA_models[i], texts=comments_lol, dictionary=id2word, coherence='c_v').get_coherence() for i in num_topics[:-1]]
        print(coherences)
        coh_sta_diffs = [coherences[i] - mean_stabilities[i] for i in range(num_keywords)[:-1]] # limit topic numbers to the number of keywords
        print(coh_sta_diffs)
        coh_sta_max = max(coh_sta_diffs)
        print(coh_sta_max)
        coh_sta_max_idxs = [i for i, j in enumerate(coh_sta_diffs) if j == coh_sta_max]
        print(coh_sta_max_idxs)
        ideal_topic_num_index = coh_sta_max_idxs[0] # choose less topics in case there's more than one max
        ideal_topic_num = num_topics[ideal_topic_num_index]
        plt.figure(figsize=(20,10))
        ax = sns.lineplot(x=num_topics[:-1], y=mean_stabilities, label='Average Topic Overlap')
        ax = sns.lineplot(x=num_topics[:-1], y=coherences, label='Topic Coherence')

        ax.axvline(x=ideal_topic_num, label='Ideal Number of Topics', color='black')
        ax.axvspan(xmin=ideal_topic_num - 1, xmax=ideal_topic_num + 1, alpha=0.5, facecolor='grey')

        y_max = max(max(mean_stabilities), max(coherences)) + (0.10 * max(max(mean_stabilities), max(coherences)))
        ax.set_ylim([0, y_max])
        ax.set_xlim([1, num_topics[-1]-1])
                        
        ax.axes.set_title('Model Metrics per Number of Topics', fontsize=25)
        ax.set_ylabel('Metric Level', fontsize=20)
        ax.set_xlabel('Number of Topics', fontsize=20)
        plt.legend(fontsize=20)
        plt.show() 

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--num_classes', metavar='num_classes', type=int,
    #                     help='num of classes in topic modelling')
    # args = parser.parse_args()
    topic_model(0)
    # optimal_topics()

# #%%
# LDAvis_prepared, doc_lda, corpus = topic_model(5)
# # %%
# topic_posts, post_topics = topic_model(5)
# # %%
