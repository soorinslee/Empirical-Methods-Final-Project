import os
import json
from matplotlib import colors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

DIR = "./data-with-sentiment"

SUM_FILE_NAME = 'summary.json'
COMMENTS_FILE_NAME = 'comments.json'
CAT_FILE_NAME = DIR+'/subsets.txt'

cats = {}
with open(CAT_FILE_NAME) as file:
    for line in file:
        temp = line.split(': ')
        cats[temp[0]] = [id.strip('\'') for id in temp[1].strip('[').strip(']\n').split(', ')]
print(cats)

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def summarize_data():
    posts = {'reddit':{}, 'youtube':{}}
    comments = {} # for now only youtube
    csv_data =[]
    # reddit_categories = ['Music', 'gaming', 'politics', 'LifeProTips']
    # youtube_categories = ['music', 'gaming', 'news', 'howto']
    csv_row = []
    last_post_hash= ''
    for _, _, files in os.walk(DIR, topdown=False):
        files.sort()
        for name in files:
            # open file
            if not name.endswith('json'):
                continue
            f = open(DIR+'/'+name,)
            data = json.load(f)
            post_num = int(name.split('-appended')[0].split('_')[-1])
            if 'views' not in data: # reddit data
                post_hash = data['post_url']
                if post_hash not in posts['reddit']:
                    post = {}
                    post['ups']=[0]*41
                    post['downs']=[0]*41
                    post['ups/downs']=[0]*41
                    post['pTop']=[0]*41
                    post['sTop']=[0]*41
                    post['pBest']=[0]*41
                    post['sBest']=[0]*41
                    post['pHot']=[0]*41
                    post['sHot']=[0]*41
                    for k, v in cats.items():
                        if post_hash in v:
                            post['cat'] = k
                    posts['reddit'][post_hash]=post
                p=0
                s=0
                if 'top_comments_top' in data:
                    for top_com in data['top_comments_top']:
                        p+=float(top_com['polarity'])
                        s+=float(top_com['subjectivity'])
                    if len(data['top_comments_top'])!=0:
                        p/=len(data['top_comments_top'])
                        s/=len(data['top_comments_top'])
                posts['reddit'][post_hash]['pTop'][post_num]=p
                posts['reddit'][post_hash]['sTop'][post_num]=s

                p=0
                s=0
                if 'top_comments_best' in data:
                    for top_com in data['top_comments_best']:
                        p+=float(top_com['polarity'])
                        s+=float(top_com['subjectivity'])
                    if len(data['top_comments_best'])!=0:
                        p/=len(data['top_comments_best'])
                        s/=len(data['top_comments_best'])
                posts['reddit'][post_hash]['pBest'][post_num]=p
                posts['reddit'][post_hash]['sBest'][post_num]=s

                p=0
                s=0
                if 'top_comments_hot' in data:
                    for top_com in data['top_comments_hot']:
                        p+=float(top_com['polarity'])
                        s+=float(top_com['subjectivity'])
                    if len(data['top_comments_hot'])!=0:
                        p/=len(data['top_comments_hot'])
                        s/=len(data['top_comments_hot'])
                posts['reddit'][post_hash]['pHot'][post_num]=p
                posts['reddit'][post_hash]['sHot'][post_num]=s
                posts['reddit'][post_hash]['ups'][post_num]=int(data['ups'])
                posts['reddit'][post_hash]['downs'][post_num]=int(data['downs'])
                if int(data['downs'])!=0:
                    posts['reddit'][post_hash]['ups/downs'][post_num]=int(data['ups'])/int(data['downs'])
            else: # youtube data
                post_hash = data['url'].split('v=')[1]
                if post_hash not in posts['youtube']:
                    if len(csv_row)!=0:
                        csv_row.extend(posts['youtube'][last_post_hash]['ups'])
                        csv_row.extend(posts['youtube'][last_post_hash]['downs'])
                        csv_row.extend(posts['youtube'][last_post_hash]['views'])
                        csv_row.extend(posts['youtube'][last_post_hash]['ups/downs'])
                        csv_row.extend(posts['youtube'][last_post_hash]['pTop'])
                        csv_row.extend(posts['youtube'][last_post_hash]['sTop'])
                        csv_row.extend(posts['youtube'][last_post_hash]['p'])
                        csv_row.extend(posts['youtube'][last_post_hash]['s'])
                        csv_data.append(csv_row)
                    csv_row = []
                    csv_row.append(post_hash)
                    post = {}
                    for k, v in cats.items():
                        if post_hash in v:
                            post['cat'] = k
                            csv_row.append(k)
                    if post['cat'] not in comments:
                        comments[post['cat']]={}
                    if post_hash not in comments[post['cat']]:
                        comments[post['cat']][post_hash]=[]
                    post['ups']=[0]*41
                    post['downs']=[0]*41
                    post['views']=[0]*41
                    post['ups/downs']=[0]*41
                    post['pTop']=[0]*41
                    post['sTop']=[0]*41
                    post['p']=[0]*41
                    post['s']=[0]*41
                    posts['youtube'][post_hash]=post
                posts['youtube'][post_hash]['ups'][post_num]=int(data['ups'])
                posts['youtube'][post_hash]['downs'][post_num]=int(data['downs'])
                posts['youtube'][post_hash]['views'][post_num]=int(data['views'])
                p=0
                s=0
                if 'top_comments' in data:
                    for top_com in data['top_comments']:
                        p+=float(top_com['polarity'])
                        s+=float(top_com['subjectivity'])
                    if len(data['top_comments'])!=0:
                        p/=len(data['top_comments'])
                        s/=len(data['top_comments'])
                    posts['youtube'][post_hash]['pTop'][post_num]=p
                    posts['youtube'][post_hash]['sTop'][post_num]=s
                p=0
                s=0
                if 'comments_since_last_sample' in data:
                    for top_com in data['comments_since_last_sample']:
                        if isEnglish(top_com['body']):
                            comments[post['cat']][post_hash].append(top_com['body'])
                        p+=float(top_com['polarity'])
                        s+=float(top_com['subjectivity'])
                    if len(data['comments_since_last_sample'])!=0:
                        p/=len(data['comments_since_last_sample'])
                        s/=len(data['comments_since_last_sample'])
                    posts['youtube'][post_hash]['p'][post_num]=p
                    posts['youtube'][post_hash]['s'][post_num]=s
                if int(data['downs'])!=0:
                    posts['youtube'][post_hash]['ups/downs'][post_num]=int(data['ups'])/int(data['downs'])
                last_post_hash= post_hash
    csv_data.append(csv_row)
    return posts, csv_data, comments

def save_summary(posts):
    with open(SUM_FILE_NAME, 'w') as f:
        json.dump(posts, f)

def save_csv(data):
    with open("out.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

def save_comments(comments):
    with open(COMMENTS_FILE_NAME, 'w') as f:
        json.dump(comments, f)

def get_summary():
    posts, csv_data, comments = summarize_data()
    # comments_cat={}
    # for k,v in cats.items():
    #     if 'Reddit' in k:
    #         continue
    #     comments_cat[k]={'max':-1, 'min':100000, 'sum':0}
    # for k,v in posts['youtube'].items():
    #     ln=len(comments[k])
    #     comments_cat[v['cat']]['sum']+=ln
    #     if ln>comments_cat[v['cat']]['max']:
    #         comments_cat[v['cat']]['max']=ln
    #     if ln<comments_cat[v['cat']]['min']:
    #         comments_cat[v['cat']]['min']=ln
    # for k,v in comments_cat.items():
    #     print(k)
    #     print('Average length {x}'.format(x=str(v['sum']/len(cats[k]))))
    #     print('Max length {x}'.format(x=str(v['max'])))
    #     print('Min length {x}'.format(x=str(v['min'])))
    save_summary(posts)
    save_csv(csv_data)
    save_comments(comments)
    return posts

if __name__ == '__main__':
    posts = get_summary()