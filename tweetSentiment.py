from transformers import pipeline
import torch, requests
import bs4

# url = "https://twitter.com/"
url = "https://twitter.com/JamesGunn/status/1659988027329089537"

# reqs = requests.get(url)
with open("jTweet.html", 'r', encoding='utf-8') as f:
    tweet = f.read()
soup = bs4.BeautifulSoup(tweet,"lxml")
# print(soup.findAll(class_="css-901oao css-16my406 r-poiln3 r-bcqeeo r-qvutc0")[0].text)


print("len of tweets found: ",str(len(soup.findAll('div',attrs={'data-testid':'tweetText'}))))
model_pth = "cardiffnlp/twitter-roberta-base-sentiment-latest"
pipe = pipeline(model=model_pth, tokenizer=model_pth, task="sentiment-analysis", device_map="auto")
def listSum(list):
    sum = 0
    for x in list:
        sum=sum+x
    return sum
def overallSent(list):
    sum = listSum(list)
    avgSent = sum/len(list)
    if avgSent > 0.2:
        return "Positive"
    elif avgSent < -0.2:
        return  "Negative"
    else:
        return "Neutral"
sentScore = []
for x in soup.findAll('div',attrs={'data-testid':'tweetText'}):
    # labels = pipe(x.text)[0]["label"]
    sentScore.append(pipe(x.text)[0]["score"])
    # print(pipe(x.text)[0]["score"])
print("The overall general sentiment is: ",overallSent(sentScore))