from sample_post import sample_yt_vid, sample_reddit_post
from time import sleep
from os import path

NUM_VIDEOS = 60
OUTPUT_DIR = "./data"
SUBREDDITS = []
YT_CATEGORIES = ['music', 'gaming', 'news', 'howto']
YT_CATEGORY_ID = {'music': 10, 'gaming': 20, 'news': 25, 'howto': 26}
SCOPES = ['https://www.googleapis.com/auth/youtube.readonly']

def youtube_authenticate():
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    api_service_name = 'youtube'
    api_version = 'v3'
    client_secrets_file = 'credentials.json'
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build(api_service_name, api_version, credentials=creds)


def most_recent_reddit_posts(subreddit):
    return []


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
        for post in reddit_urls:
            output_filename = path.join(OUTPUT_DIR, f"{hash(post)}_{i}")
            try:
                sample_reddit_post(post, output_filename)
            except Exception as e:
                print(e)
        for post in youtube_video_ids:
            output_filename = path.join(OUTPUT_DIR, f"{hash(post)}_{i}")
            try:
                sample_yt_vid(post, output_filename)
            except Exception as e:
                print(e)
        i += 1
        sleep(60*60*4)


if __name__ == '__main__':
    main()
