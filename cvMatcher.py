# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nlp import nlp as nlp
from collections import Counter
from fuzzywuzzy import fuzz
import Levenshtein as lev
import spacy
Spnlp = spacy.load("en_core_web_sm")
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(Spnlp.vocab)
import matplotlib.pyplot as plt
from wordcloud import WordCloud

LangProcessor = nlp()


# %%
#load the job description

with open('identity.txt') as job:
    text = job.read()    


# %%
#load cv

with open('cv') as cv:
    cvtext = cv.read()


# %%
jobContent = ""
for t in text:
    jobContent = jobContent + t.lower().replace("'",'')

cvContent = ""
for c in cvtext:
    cvContent = cvContent + c.lower().replace("'",'')


# %%
keywordsJob = LangProcessor.keywords(jobContent)
keywordsCV = LangProcessor.keywords(cvContent)


# %%
#print(keywordsJob['ranked phrases'][:10])

for item in keywordsJob['ranked phrases'][:10]:
    print (str(round(item[0],2)) + ' - ' + item[1] )


# %%
for item in keywordsCV['ranked phrases'][:10]:
    print (str(round(item[0],2)) + ' - ' + item[1] )


# %%

#from https://medium.com/@adriensieg/text-similarities-da019229c894
def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)



sims = []
phrases = []
for key in keywordsJob['ranked phrases']:
    rec={}
    rec['importance'] = key[0]
    texts = key[1]
    sims=[]
    avg_sim=0
    for cvkey in keywordsCV['ranked phrases']:
        cvtext = cvkey[1]
        sims.append(fuzz.ratio(texts, cvtext))
        #sims.append(lev.ratio(texts.lower(),cvtext.lower()))
        #sims.append(jaccard_similarity(texts,cvtext))
        
    count=0
    for s in sims:
        count=count+s

    avg_sim = count/len(sims)    
    rec['similarity'] = avg_sim
    rec['text'] = texts
    phrases.append(rec)


# %%
tokensJob = LangProcessor.tokenize(jobContent)


# %%
tokensCv = LangProcessor.tokenize(cvContent)


# %%
job = Counter(tokensJob).most_common(20)

# lower max_font_size
wordcloud = WordCloud(max_font_size=40).generate(text)
plt.figure(figsize=(14,14))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# %%
wordcloud = WordCloud(max_font_size=40).generate(cvContent)
plt.figure(figsize=(14,14))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# %%
cv = Counter(tokensCv).most_common(20)


# %%
data = []
for r in job:
    rec = {}
    rec['word'] = r[0]
    rec['from'] = 'job'
    rec['freq'] = r[1]
    data.append(rec)

for r in cv:
    rec = {}
    rec['word'] = r[0]
    rec['from'] = 'cv'
    rec['freq'] = r[1]
    data.append(rec)


# %%
import pandas as pd
df=pd.DataFrame(data)
pu=pd.DataFrame(phrases)

ph = pd.melt(pu[:10], id_vars=['text'], value_vars=['importance','similarity'])
pu.head(10)


# %%
keyPhrases = pu['text'].values
print(keyPhrases)


# %%
terms = keyPhrases
# Only run nlp.make_doc to speed things up
patterns = [Spnlp.make_doc(t) for t in terms]
matcher.add("Spec", patterns)


# %%
doc = Spnlp(cvContent)
matchkeywords = []
matches = matcher(doc)
for match_id, start, end in matches:
    span = doc[start:end]
    matchkeywords.append(span.text)


# %%
a = Counter(matchkeywords)
print(a)


# %%
vals = []
for key, val in a.items():
    rec = {}
    imp = pu[pu['text'] == key]['importance'].values[0]
    rec['word'] = key
    rec['importance'] = imp
    rec['influence'] = val 
    vals.append(rec) 

rData = pd.DataFrame(vals)
rData.head()


# %%
import altair as alt


# %%
base = alt.Chart(rData).mark_point().encode(
    x='influence:Q',
    y='word:N',
).properties(
    width=400,
    height=400
)

alt.vconcat(
   base.encode(color='importance:Q').properties(title='Job spec key phrases cross reference CV')
)


# %%
d = (alt.
    Chart(ph).
    mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).
    encode(y='text', x='value', color='variable').
    properties(height=300, width=700, title='Key phrase match')  
)

dasc = d.encode(alt.Y(field='text', type='nominal', sort='-x'))


# %%
p = (alt.
    Chart(df).
    mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).
    encode(x='freq',y='word', color='from:N').
    properties(height=300, width=700, title='Words in Job and CV')    
    
)

pasc = p.encode(alt.Y(field='word', type='nominal', sort='-x'))


# %%
dasc


# %%
pasc


# %%



