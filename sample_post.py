import requests
from utils import youtube_authenticate
import pprint
import datetime
import json


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


def sample_yt_vid(video_id, output_file):
    """
    Collects the relevant info and dumps it into output_file (JSON format)
    """
    sample_data = {}
    youtube = youtube_authenticate()
    videos_response = youtube.videos().list(
        part='statistics,snippet',
        id=video_id
    ).execute()
    sample_data['time_posted'] = videos_response['items'][0]['snippet']['publishedAt']
    sample_data['url'] = f"https://www.youtube.com/watch?v={video_id}"
    sample_data['time_now'] = datetime.datetime.utcnow().isoformat()
    sample_data['ups'] = videos_response['items'][0]['statistics']['likeCount']
    sample_data['downs'] = videos_response['items'][0]['statistics']['dislikeCount']
    sample_data['views'] = videos_response['items'][0]['statistics']['viewCount']
    comments_response = youtube.commentThreads().list(
        part="snippet",  # add REPLIES to also get 2nd-level replies
        videoId=video_id,
    ).execute()
    pprint.pprint(comments_response)
    with open(output_file, "w") as f:
        f.write("JSON goes here")
