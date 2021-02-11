# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nlp import nlp as nlp

LangProcessor = nlp()


# %%
with open("anal.pickle", "rb") as f:
    data = pickle.load(f)


corpus = []
cols = []

for d in data:
    corpus.append(d['text'])

for d in data:
    cols.append(d['link'])    


# %%
vectorizer = TfidfVectorizer(stop_words='english', ngram_range = (1,1), max_df = .6, min_df = .01, decode_error='ignore', strip_accents='unicode', analyzer='word',
    tokenizer=LangProcessor.tokenize)


# %%
X = vectorizer.fit_transform(corpus)


# %%
feature_names = vectorizer.get_feature_names()


# %%
dense = X.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
df.head()


# %%
dat = df.transpose()

recs = []

# Find the top 30 words written by each author
top_dict = {}
for c in range(0,len(cols)):
    top = dat.iloc[:,c].sort_values(ascending=False).head(30)
    top_dict[dat.columns[c]]= list(zip(top.index, top.values))# Print the top 15 words from each article
for p, top_words in top_dict.items():
    rec = {'link': cols[p]}
    index = 0
    for word, count in top_words[0:5]:
        rec[str(index)+" term"] = word
        rec[str(index)+" termCounts"] = count
        index = index+1

    stats = LangProcessor.stats(data[p]['text'])
    sentiment = LangProcessor.lang_senti(data[p]['text'])
    rec['standard'] = stats['standard']
    rec['lexCount'] = stats['lexicon count'] 
    rec['flesch'] = stats['flesch_reading_ease']
    rec['pol'] = sentiment[0]
    rec['subjectivity'] = sentiment[1]
    recs.append(rec)

print(recs[0])


# %%
with open("analytis.pickle", "wb") as f:
    pickle.dump(recs, f, pickle.HIGHEST_PROTOCOL)


# %%
dataframe = pd.DataFrame(recs)


# %%
dataframe.head()


# %%
dataframe.info()


# %%
import altair as alt

grades = {
'10th and 11th grade': '4: 10th-11th',
'11th and 12th grade': '5: 11th-12th',
'12th and 13th grade': '6: 12th-13th',
'13th and 14th grade': '7: 13th-14th',
'14th and 15th grade': '8: 14th-15th',
'15th and 16th grade': '9: 15th-16th',
'16th and 17th grade': '10: 16th-17th',  
'17th and 18th grade': '11: > 17th',  
'18th and 19th grade': '11: > 17th',  
'19th and 20th grade': '11: > 17th',  
'20th and 21st grade': '11: > 17th',  
'21st and 22nd grade': '11: > 17th', 
'22nd and 23rd grade': '11: > 17th',  
'23rd and 24th grade': '11: > 17th',  
'24th and 25th grade': '11: > 17th',  
'25th and 26th grade': '11: > 17th',  
'26th and 27th grade': '11: > 17th',
'27th and 28th grade': '11: > 17th', 
'28th and 29th grade': '11: > 17th',
'29th and 30th grade': '11: > 17th',
'30th and 31st grade': '11: > 17th', 
'31st and 32nd grade': '11: > 17th',
'32nd and 33rd grade': '11: > 17th', 
'35th and 36th grade': '11: > 17th',  
'36th and 37th grade': '11: > 17th',
'37th and 38th grade': '11: > 17th', 
'38th and 39th grade': '11: > 17th',
'3rd and 4th grade':   '1: 3rd-4th', 
'57th and 58th grade': '11: > 17th',
'8th and 9th grade':   '2: 8th-9th',
'9th and 10th grade':  '3: 9th-10th'}

def flesch(x):
    response = 'Extremely confusing'
    if x< 30:
        response = 'Very confusing'
    elif x>29 and x<50:
        response = 'Difficult'
    elif x>49 and x<60:
        response = 'Fairly Difficult'

    elif x>59 and x<70:
        response = 'Standard'

    elif x>69 and x<80:
        response = 'Fairly Easy'

    elif x>79 and x<90:
        response = 'Easy'

    elif x > 89:
        response = 'Very Easy'
 
    return response


dataframe['ReadingDiff'] = dataframe.flesch.apply(lambda x: flesch(x))

f = dataframe.groupby('standard').count()['link']
f = f.reset_index()
f.columns = ['Grade','Count']

f['cat']= f.Grade.apply(lambda x: grades[x])

f


# %%
dataframe['lexCount'].plot()


# %%
p = (alt.
    Chart(dataframe).
    mark_circle(size=40).
    encode(x='pol',y='subjectivity').
    properties(height=200, width=400, title='objectivity versus polarity')    
    
)

p1Line = alt.Chart(dataframe).mark_rule(color='red').encode(y='mean(subjectivity):Q')
p2Line = alt.Chart(dataframe).mark_rule(color='red').encode(x='mean(pol):Q')

line = pd.DataFrame({'x': [0,0], 'y': [0,1]})
p3Line = alt.Chart(line).mark_line(color='green').encode(x='x',y='y')

p3 = p+p1Line+p2Line+p3Line


# %%
p1Line = alt.Chart(dataframe).mark_rule(color='red').encode(y='mean(flesch):Q')
p2Line = alt.Chart(dataframe).mark_rule(color='red').encode(x='mean(lexCount):Q')

p = (alt.
    Chart(dataframe).
    mark_circle(size=40).
    encode(x='lexCount', y='flesch').
    properties(height=200, width=400, title='Article length versus readability').
    interactive()

)

p1 = p+p1Line+p2Line


# %%
p2 = (alt.
    Chart(f).
    mark_bar().
    encode(x='cat',y='sum(Count):Q').
    properties(height=200, width=400, title='Standard reading score')
)

p2asc = p2.encode(alt.X(field='cat', type='nominal', sort='y'))

p4 = (alt.
    Chart(dataframe).
    mark_bar().
    encode(x = 'ReadingDiff', y='count()').
    properties(height=200, width=400, title='Reading level')
)

p4asc = p4.encode(alt.X(field='ReadingDiff', type='nominal', sort='y'))


# %%
p1 & p2asc | (p3 & p4asc)


# %%
flesch(-9), flesch(109)


# %%



