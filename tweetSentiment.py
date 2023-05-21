from transformers import pipeline
import torch, time
from splinter import Browser
from selenium import webdriver
def listSum(list):
    sum = 0
    for x in list:
        sum+=x
    return sum
def overallSentScore(list):
    sum = listSum(list)
    avgSent = sum/len(list)
    if avgSent > 0.2:
        return "Positive"
    elif avgSent < -0.2:
        return "Negative"
    else:
        return "Neutral"
def overallSent(dict):
    sum = (dict['positive'] - dict['negative']) + (0.5*dict['neutral'])
    if sum > 0:
        return "Positive"
    elif sum < 0:
        return "Negative"
def oldWay():
    import bs4, requests
    model_pth = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    pipe = pipeline(model=model_pth, tokenizer=model_pth, task="sentiment-analysis", device_map="auto")
    with open("jTweet.html", 'r', encoding='utf-8') as f:
        tweet = f.read()
    soup = bs4.BeautifulSoup(tweet, "lxml")

    print("len of tweets found: ", str(len(soup.findAll('div', attrs={'data-testid': 'tweetText'}))))

    sentScore = []
    for x in soup.findAll('div', attrs={'data-testid': 'tweetText'}):
        # labels = pipe(x.text)[0]["label"]
        sentScore.append(pipe(x.text)[0]["score"])
        # print(pipe(x.text)[0]["score"])
    print("The overall general sentiment is: ", overallSent(sentScore))
# url = "https://twitter.com/"
# url = "https://twitter.com/JamesGunn/status/1659988027329089537"
url = "https://twitter.com/KamalaHarris/status/1658559607239766020"
# reqs = requests.get(url)

tweets = []
firefox = Browser('firefox')
# firefox = webdriver.Firefox()
firefox.visit(url)
time.sleep(1)
# firefox.execute_script('document.body.style.zoom = "25%"')
tweetCounter = 0
for x in range(10):
    for tweetGroup in firefox.find_by_css('[data-testid="tweetText"]'):
        tweetCounter += 1
        # print(tweetGroup.text)
        tweets.append(tweetGroup.text)
    firefox.execute_script('window.scrollTo(0, document.body.scrollHeight);')
    time.sleep(0.5)

print("Length of tweets found:", str(tweetCounter))
model_pth = "cardiffnlp/twitter-roberta-base-sentiment-latest"
pipe = pipeline(model=model_pth, tokenizer=model_pth, task="sentiment-analysis", device_map="auto")
scoreList = {
    "positive": 0,
    "neutral": 0,
    "negative":0
}
for tweet in tweets:
    scoreList[(pipe(tweet)[0]['label'])] += 1
    # print(pipe(tweet)[0]['score'])

print("The general sentiment of this tweet is:", overallSent(scoreList))