def sample_reddit_post(url, output_file):
    """
    :param url: post URL
    Collects the relevant info and dumps it into output_file (JSON format)
    """
    heuristics = ["top", "best", "hot"]
    for heur in heuristics:
        updated_url = url+'.json?sort='+heur
        response = requests.get(updated_url, headers={'User-agent': 'bot'})
    with open(output_file, "w") as f:
        f.write("JSON goes here")


def sample_yt_vid(url, output_file):
    """
    :param url: post URL
    Collects the relevant info and dumps it into output_file (JSON format)
    """
    with open(output_file, "w") as f:
        f.write("JSON goes here")
