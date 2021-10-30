import requests
from utils import youtube_authenticate
import pprint
import datetime
import json
import numpy as np

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
HEURISTICS = ["top", "best", "hot"]


def sample_reddit_post(url, output_file, last_output_filename):
    """
    Collects the relevant info and dumps it into output_file (JSON format)
    """

    def reformat_datetime(dt):
        return datetime.datetime.fromtimestamp(dt).strftime(DATETIME_FORMAT)

    sample_data = {}
    updated_url = url+'/.json?sort=new&limit=1000'
    response = requests.get(updated_url, headers={'User-agent': 'bot'})
    response = response.json()
    sample_data['post_url'] = url
    post_info = response[0]['data']['children'][0]['data']

    sample_data['time_posted'] = reformat_datetime(post_info['created_utc'])
    sample_data['time_now'] = datetime.datetime.utcnow().strftime(DATETIME_FORMAT)
    sample_data['ups'] = post_info['ups']
    sample_data['downs'] = post_info['downs']

    # Get the time at which the last sample was taken for this post (if applicable)
    last_sample_time = None
    if last_output_filename is not None:
        with open(last_output_filename, "r") as f:
            last_sample_time = datetime.datetime.strptime(
                json.load(f)['time_now'],
                DATETIME_FORMAT
            )

    new_comments=[]
    for comment in response[1]['data']['children']:
        if 'created_utc' not in comment['data']: # last object would contain IDs of other comments, we skip it for now
            continue
        comment_time = reformat_datetime(comment['data']['created_utc'])
        if last_sample_time is None or comment_time > last_sample_time:
            new_comments.append(
                {'comment_id': comment['data']['id'], 'body': comment['data']['body']}
            )
    sample_data['comments_since_last_sample']=new_comments
    for heur in HEURISTICS:
        heur_comments=[]
        updated_url = url+'/.json?sort='+heur
        response = requests.get(updated_url, headers={'User-agent': 'bot'})
        response = response.json()
        for comment in response[1]['data']['children']:
            if len(heur_comments)<5:
                heur_comments.append(
                    {'comment_id': comment['data']['id'], 'body': comment['data']['body'], 'ups': comment['data']['ups'], 'downs': comment['data']['downs']}
                )
        sample_data['top_3_comments_'+heur]=heur_comments
    with open(output_file, "w") as f:
        json.dump(sample_data, f, indent=4)


def sample_yt_vid(video_id, output_file, last_output_filename):
    """
    Collects the relevant info and dumps it into output_file (JSON format)
    """

    def reformat_datetime(dt):
        return dt.replace("Z", "").replace("T", " ")

    sample_data = {}
    youtube = youtube_authenticate()
    videos_response = youtube.videos().list(
        part='statistics,snippet',
        id=video_id
    ).execute()
    sample_data['time_posted'] = reformat_datetime(videos_response['items'][0]['snippet']['publishedAt'])
    sample_data['url'] = f"https://www.youtube.com/watch?v={video_id}"
    sample_data['time_now'] = datetime.datetime.utcnow().strftime(DATETIME_FORMAT)
    sample_data['ups'] = videos_response['items'][0]['statistics']['likeCount']
    sample_data['downs'] = videos_response['items'][0]['statistics']['dislikeCount']
    sample_data['views'] = videos_response['items'][0]['statistics']['viewCount']

    comments_response = youtube.commentThreads().list(
        part="snippet",  # change to "snippet,replies" to also get 2nd-level replies
        videoId=video_id,
    ).execute()

    # Get the time at which the last sample was taken for this post (if applicable)
    last_sample_time = None
    if last_output_filename is not None:
        with open(last_output_filename, "r") as f:
            last_sample_time = datetime.datetime.strptime(
                json.load(f)['time_now'],
                DATETIME_FORMAT
            )

    new_comments = []
    comments, n_likes, reply_count = [], [], []
    for item in comments_response['items']:
        comment_info = item['snippet']['topLevelComment']['snippet']
        comment_time = datetime.datetime.strptime(
            reformat_datetime(comment_info['publishedAt']),
            DATETIME_FORMAT
        )
        if last_sample_time is None or comment_time > last_sample_time:
            new_comments.append(comment_info["textOriginal"])
        comments.append(comment_info["textOriginal"])
        n_likes.append(comment_info["likeCount"])
        reply_count.append(item["snippet"]["totalReplyCount"])
    k = min(len(comments), 5)
    if k > 0:
        top_k_indices = np.argsort(n_likes)[-k:]
        top_comments = []
        for index in top_k_indices:
            if n_likes[index] > 0:
                top_comments.append(
                    {"comment": comments[index], "likes": n_likes[index], "number_of_replies": reply_count[index]}
                )
        sample_data["top_comments"] = top_comments
    sample_data["comments_since_last_sample"] = [{"body": new_comment} for new_comment in new_comments]

    with open(output_file, "w") as f:
        json.dump(sample_data, f, indent=4)
