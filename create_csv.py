import csv
import os
import glob
import json
from datetime import datetime


# calculate time since posted in hours
def get_time_since_posted(time_posted, time_now):
    date_format_str = '%Y-%m-%d %H:%M:%S'
    start = datetime.strptime(time_posted, date_format_str)
    end = datetime.strptime(time_now, date_format_str)
    diff = end - start
    diff_in_hours = diff.total_seconds() / 3600
    return diff_in_hours


# get average polarity and subjectivity of all comments
def get_all_avgs(comments, prev_avg_polarity, prev_avg_subjectivity, prev_num_comments):
    sum_polarity = 0.0
    sum_subjectivity = 0.0
    curr_num_comments = 0
    for comment in comments:
        if comment['polarity'] != 0.0 and comment['subjectivity'] != 0.0:
            curr_num_comments += 1
            sum_polarity += comment['polarity']
            sum_subjectivity += comment['subjectivity']
    if curr_num_comments != 0:
        all_avg_polarity = (prev_avg_polarity * prev_num_comments + sum_polarity) / (prev_num_comments +
                                                                                     curr_num_comments)
        all_avg_subjectivity = (prev_avg_subjectivity * prev_num_comments + sum_subjectivity) / (prev_num_comments +
                                                                                                 curr_num_comments)
        num_comments = prev_num_comments + curr_num_comments
    else:
        all_avg_polarity = prev_avg_polarity
        all_avg_subjectivity = prev_avg_subjectivity
        num_comments = prev_num_comments
    return all_avg_polarity, all_avg_subjectivity, num_comments


# get average polarity and subjectivity of top comments for Reddit
def get_top_avgs_reddit(comments):
    sum_polarity_wt = 0
    sum_subjectivity_wt = 0
    sum_polarity = 0.0
    sum_subjectivity = 0.0
    for comment in comments:
        if comment['polarity'] != 0.0:
            sum_polarity_wt += comment['ups'] - comment['downs']
            sum_polarity += (comment['ups'] - comment['downs']) * comment['polarity']
        if comment['subjectivity'] != 0.0:
            sum_subjectivity_wt += comment['ups'] - comment['downs']
            sum_subjectivity += (comment['ups'] - comment['downs']) * comment['subjectivity']
    if sum_polarity_wt != 0:
        top_avg_polarity = sum_polarity / sum_polarity_wt
    else:
        top_avg_polarity = ''
    if sum_subjectivity_wt != 0:
        top_avg_subjectivity = sum_subjectivity / sum_subjectivity_wt
    else:
        top_avg_subjectivity = ''
    return top_avg_polarity, top_avg_subjectivity


# get average polarity and subjectivity of top comments for YouTube
def get_top_avgs_yt(comments):
    sum_polarity_wt = 0
    sum_subjectivity_wt = 0
    sum_polarity = 0.0
    sum_subjectivity = 0.0
    for comment in comments:
        if comment['polarity'] != 0.0:
            sum_polarity_wt += comment['likes']
            sum_polarity += comment['likes'] * comment['polarity']
        if comment['subjectivity'] != 0.0:
            sum_subjectivity_wt += comment['likes']
            sum_subjectivity += comment['likes'] * comment['subjectivity']
    if sum_polarity_wt != 0:
        top_avg_polarity = sum_polarity / sum_polarity_wt
    else:
        top_avg_polarity = ''
    if sum_subjectivity_wt != 0:
        top_avg_subjectivity = sum_subjectivity / sum_subjectivity_wt
    else:
        top_avg_subjectivity = ''
    return top_avg_polarity, top_avg_subjectivity


# store categories as lists of urls
subsets = open('data-with-sentiment/subsets.txt', 'r')
for line in subsets:
    line = line.strip().replace(' ', '').replace('\'', '')
    urls = line[line.find('[') + 1:line.find(']')]
    if 'Reddit-Music' in line:
        reddit_music = urls.split(',')
    elif 'Reddit-gaming' in line:
        reddit_gaming = urls.split(',')
    elif 'Reddit-politics' in line:
        reddit_politics = urls.split(',')
    elif 'Reddit-LifeProTips' in line:
        reddit_lifeprotips = urls.split(',')
    elif 'YouTube-music' in line:
        youtube_music = urls.split(',')
    elif 'YouTube-gaming' in line:
        youtube_gaming = urls.split(',')
    elif 'YouTube-news' in line:
        youtube_news = urls.split(',')
    elif 'YouTube-howto' in line:
        youtube_howto = urls.split(',')
subsets.close()

with open('samples.csv', 'w', newline='') as csvfile:
    fieldnames = ['category', 'url', 'time_since_posted', 'ups', 'downs', 'views', 'num_comments', 'all_avg_polarity',
                  'all_avg_subjectivity', 'top_avg_polarity', 'top_avg_subjectivity']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='')
    writer.writeheader()

    # iterate through JSON files in folder
    path = 'data-with-sentiment'
    prev_avg_polarity = 1.0
    prev_avg_subjectivity = 1.0
    prev_num_comments = 0
    for filename in glob.glob(os.path.join(path, '*.json')):
        with open(filename, 'r') as json_file:
            data = json_file.read().replace('\n', '')
            data_dict = json.loads(data)
            ups = data_dict['ups']
            downs = data_dict['downs']
            time_posted = data_dict['time_posted']
            time_now = data_dict['time_now']
            time_since_posted = get_time_since_posted(time_posted, time_now)
            if 'comments_since_last_sample' in data_dict:
                all_avg_polarity, all_avg_subjectivity, num_comments = get_all_avgs(
                    data_dict['comments_since_last_sample'], prev_avg_polarity, prev_avg_subjectivity,
                    prev_num_comments)
                prev_avg_polarity = all_avg_polarity
                prev_avg_subjectivity = all_avg_subjectivity
                prev_num_comments = num_comments
            else:
                all_avg_polarity = prev_avg_polarity
                all_avg_subjectivity = prev_avg_subjectivity
                num_comments = prev_num_comments

            # sample is from Reddit
            if 'post_url' in data:
                url = data_dict['post_url']
                # get category
                if url in reddit_music:
                    category = 'Reddit_Music'
                elif url in reddit_gaming:
                    category = 'Reddit_gaming'
                elif url in reddit_politics:
                    category = 'Reddit_politics'
                elif url in reddit_lifeprotips:
                    category = 'Reddit_LifeProTips'
                # TOP heuristic
                if 'top_comments_top' in data_dict:
                    top_polarity, top_subjectivity = get_top_avgs_reddit(data_dict['top_comments_top'])
                else:
                    top_polarity = ''
                    top_subjectivity = ''

                # BEST heuristic
                if 'top_comments_best' in data_dict:
                    best_polarity, best_subjectivity = get_top_avgs_reddit(data_dict['top_comments_best'])
                else:
                    best_polarity = ''
                    best_subjectivity = ''
                # HOT heuristic
                if 'top_comments_hot' in data_dict:
                    hot_polarity, hot_subjectivity = get_top_avgs_reddit(data_dict['top_comments_hot'])
                else:
                    hot_polarity = ''
                    hot_subjectivity = ''
                top_avg_polarity = [top_polarity, best_polarity, hot_polarity]
                top_avg_subjectivity = [top_subjectivity, best_subjectivity, hot_subjectivity]
                writer.writerow({'category': category, 'url': url, 'time_since_posted': time_since_posted, 'ups': ups,
                                 'downs': downs, 'num_comments': num_comments, 'all_avg_polarity': all_avg_polarity,
                                 'all_avg_subjectivity': all_avg_subjectivity, 'top_avg_polarity': top_avg_polarity,
                                 'top_avg_subjectivity': top_avg_subjectivity})

            # sample is from YouTube
            elif 'url' in data:
                url = data_dict['url']
                views = data_dict['views']
                # find ID from url
                id = url.split('https://www.youtube.com/watch?v=', 1)[1]
                # get category
                if id in youtube_music:
                    category = 'YouTube_music'
                elif id in youtube_gaming:
                    category = 'YouTube_gaming'
                elif id in youtube_news:
                    category = 'YouTube_news'
                elif id in youtube_howto:
                    category = 'YouTube_howto'
                if 'top_comments' in data_dict:
                    top_avg_polarity, top_avg_subjectivity = get_top_avgs_yt(data_dict['top_comments'])
                else:
                    top_avg_polarity = ''
                    top_avg_subjectivity = ''
                writer.writerow({'category': category, 'url': url, 'time_since_posted': time_since_posted, 'ups': ups,
                                 'downs': downs, 'views': views, 'num_comments': num_comments,
                                 'all_avg_polarity': all_avg_polarity, 'all_avg_subjectivity': all_avg_subjectivity,
                                 'top_avg_polarity': top_avg_polarity, 'top_avg_subjectivity': top_avg_subjectivity})
        json_file.close()
csvfile.close()
