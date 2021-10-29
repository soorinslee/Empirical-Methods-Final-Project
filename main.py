from sample_post import sample_yt_vid, sample_reddit_post
from time import sleep
from os import path
import requests
from utils import youtube_authenticate
import os
import shutil

NUM_VIDEOS = 50
NUM_SUBREDDITS = 100
OUTPUT_DIR = './data'
SUBREDDITS = ['Music', 'gaming', 'politics', 'LifeProTips']
YT_CATEGORIES = ['music', 'gaming', 'news', 'howto']
YT_CATEGORY_ID = {'music': 10, 'gaming': 20, 'news': 25, 'howto': 26}


def most_recent_reddit_posts(subreddit):
    limit = NUM_SUBREDDITS
    timeframe = 'all'
    listing = 'new'
    try:
        base_url = f'https://www.reddit.com/r/{subreddit}/{listing}.json?limit={limit}&t={timeframe}'
        response = requests.get(base_url, headers={'User-agent': 'bot'})
    except:
        print('An Error Occured')
    posts = response.json()['data']['children']
    urls = []
    for post in posts:
        urls.append('https://www.reddit.com' + post['data']['permalink'])

    return urls


def most_recent_youtube_vids(category):
    youtube = youtube_authenticate()
    request = youtube.search().list(
        part='snippet',
        type='video',
        order='date',
        videoCategoryId=YT_CATEGORY_ID[category],
        relevanceLanguage='en',
        regionCode='US',
        maxResults=NUM_VIDEOS
    )
    response = request.execute()
    video_ids = []
    for video in response['items']:
        video_ids.append(video['id']['videoId'])

    return video_ids


def main():
    reddit_urls, youtube_video_ids = [], []
    for sr in SUBREDDITS:
        reddit_urls.extend(most_recent_reddit_posts(sr))
    for cat in YT_CATEGORIES:
        youtube_video_ids.extend(most_recent_youtube_vids(cat))
    i = 0
    while True:
        for post in youtube_video_ids:
            output_filename = path.join(OUTPUT_DIR, f'{hash(post)}_{i}.json')
            last_output_filename = None
            if i > 0:
                last_output_filename = path.join(OUTPUT_DIR, f'{hash(post)}_{i}.json')
            try:
                sample_yt_vid(post, output_filename, last_output_filename)
            except Exception as e:
                print(e)
        for post in reddit_urls:
            output_filename = path.join(OUTPUT_DIR, f'{hash(post)}_{i}.json')
            try:
                sample_reddit_post(post, output_filename)
            except Exception as e:
                print(e)
        i += 1
        sleep(60*60*4)


if __name__ == '__main__':
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)

    main()
