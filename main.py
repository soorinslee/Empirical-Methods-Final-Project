from sample_post import sample_yt_vid, sample_reddit_post
from time import sleep
from os import path
import requests
from utils import youtube_authenticate
import os
import shutil
import traceback

VIDS_PER_CAT = 50
POSTS_PER_SUBRED = 100
OUTPUT_DIR = './data'
SUBREDDITS = ['Music', 'gaming', 'politics', 'LifeProTips']
YT_CATEGORIES = ['music', 'gaming', 'news', 'howto']
YT_CATEGORY_ID = {'music': 10, 'gaming': 20, 'news': 25, 'howto': 26}


def most_recent_reddit_posts(subreddit):
    limit = POSTS_PER_SUBRED
    timeframe = 'all'
    listing = 'new'
    base_url = f'https://www.reddit.com/r/{subreddit}/{listing}.json?limit={limit}&t={timeframe}'
    response = requests.get(base_url, headers={'User-agent': 'bot'})
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
        maxResults=VIDS_PER_CAT
    )
    response = request.execute()
    video_ids = []
    for video in response['items']:
        video_ids.append(video['id']['videoId'])

    return video_ids


def main():
    reddit_urls, youtube_video_ids = [], []
    with open(os.path.join(OUTPUT_DIR, "subsets.txt"), "w") as subset_file:
        for sr in SUBREDDITS:
            most_recent = most_recent_reddit_posts(sr)
            reddit_urls.extend(most_recent)
            subset_file.write(f"Reddit - {sr}: {most_recent}\n")
        for cat in YT_CATEGORIES:
            most_recent = most_recent_youtube_vids(cat)
            youtube_video_ids.extend(most_recent)
            subset_file.write(f"YouTube - {cat}: {most_recent}\n")
    i = 0
    while True:
        for post in youtube_video_ids:
            output_filename = path.join(OUTPUT_DIR, f'{post}_{i}.json')
            last_output_filename = None
            if i > 0:
                last_output_filename = path.join(OUTPUT_DIR, f'{post}_{i-1}.json')
            try:
                sample_yt_vid(post, output_filename, last_output_filename)
            except Exception as e:
                print(e)
                traceback.print_exc()
        for post in reddit_urls:
            output_filename = path.join(OUTPUT_DIR, f'{hash(post)}_{i}.json')
            last_output_filename = None
            if i > 0:
                last_output_filename = path.join(OUTPUT_DIR, f'{hash(post)}_{i-1}.json')
            try:
                sample_reddit_post(post, output_filename, last_output_filename)
            except Exception as e:
                print(e)
                traceback.print_exc()
        i += 1
        sleep(60*60*4)


if __name__ == '__main__':
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)

    main()
