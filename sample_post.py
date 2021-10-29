import requests
from utils import youtube_authenticate
import pprint
import datetime
import json
import numpy as np

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def sample_reddit_post(url, output_file):
    """
    Collects the relevant info and dumps it into output_file (JSON format)
    """
    heuristics = ["top", "best", "hot"]
    for heur in heuristics:
        updated_url = url+'.json?sort='+heur
        response = requests.get(updated_url, headers={'User-agent': 'bot'})
    with open(output_file, "w") as f:
        f.write("JSON goes here")


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
