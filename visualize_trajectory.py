from trajectory import load_data
from csv import DictReader
import matplotlib.pyplot as plt

THRESHOLD = .0

with open("samples.csv", "r") as f:
    reader = DictReader(f)
    urls = set()
    for row in reader:
        #if row["category"] == "YouTube_news":
        if "YouTube" in row["category"]:
        #if "Reddit" in row["category"]:
            urls.add(row["url"])


if __name__ == "__main__":
    for url in urls:
        X, Y = load_data(urls=[url], normalize=False, include_avg_sentiment=False,
                         include_weighted_avg_top_sentiment=False)
        if len(X) == 0:
            continue
        #time = [0]
        time = [X[0][2]]
        avg_rating = [X[0][3]]
        if avg_rating[0] < THRESHOLD:
            continue
        for i in range(len(X)):
            avg_rating.append(avg_rating[i] + Y[i]*4)
            time.append((i+1)*4)
        plt.plot(time, avg_rating, linewidth=5, alpha=.2, color='red')
    plt.show()

