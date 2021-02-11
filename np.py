import time
import nltk
from bs4 import BeautifulSoup
from bs4.element import Comment
import re
import pickle


from selenium import webdriver
driver = webdriver.Firefox()
driver.get("get your own login token") 
print('signing in buddy')
time.sleep(5)

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    if re.match(r"[\n]+",str(element)): return False
    return True

def text_from_html(html):
    body = html
    soup = BeautifulSoup(body ,"lxml")
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    text = u",".join(t.strip() for t in visible_texts)
    text = text.lstrip().rstrip()
    text = text.split(',')
    clean_text = ''
    for sen in text:
        if sen:
            sen = sen.rstrip().lstrip()
            clean_text += sen+','
    return clean_text




with open('urls.txt', 'r') as f:
    url = f.readlines()

clean_urls = []
for uri in url:
    print(uri)
    if ("?source=" in uri):
        print('bad')
    else:
        clean_urls.append(uri)

print(len(clean_urls))

#print(clean_urls)
texts = []

for url in clean_urls:
    record  = {}
    driver.get(url)
    time.sleep(2)
    html = driver.page_source

    tx = text_from_html(html)
    record = {'link': url, 'text': tx}
    texts.append(record)
    time.sleep(20)

driver.close()

print(texts)

with open("anal.pickle", "wb") as f:
    pickle.dump(texts, f, pickle.HIGHEST_PROTOCOL)