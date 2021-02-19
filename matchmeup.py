# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Import summarize from gensim
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords 
# Import the library# to convert MSword doc to txt for processing.
import docx2txt
import spacy
Spnlp = spacy.load("en_core_web_sm")
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(Spnlp.vocab)
from collections import Counter
import pandas as pd
import PyPDF4
import re
import io


# %%

#https://towardsdatascience.com/do-the-keywords-in-your-resume-aptly-represent-what-type-of-data-scientist-you-are-59134105ba0d
def pdfextract(file):
    pdfFileObj = open(file, 'rb')
    pdfReader = PyPDF4.PdfFileReader(pdfFileObj)
    countpage = pdfReader.getNumPages()
    count = 0
    text = []
    while count < countpage:    
        pageObj = pdfReader.getPage(count)
        count +=1
        t = pageObj.extractText()
        #for line in t.split('\n'):
        #    #if re.match(r"^PDF", line):
        #    print(line)
        text.append(t)
    return text

import pdfplumber
def pdfText(file):
    with pdfplumber.open(file) as pdf:
        first_page= pdf.pages[0]
        second_page = pdf.pages[1]
        #print(first_page, second_page)
        return [first_page.extract_text(), second_page.extract_text()]

import pdftotext
def textFromPdf(file):
    with open(file, 'r', encoding='Latin-1') as f:
        pdf = pdftotext.PDF(f)

    # Iterate over all the pages
    for page in pdf:
        print(page)   

    return 'start'


# %%
import docx2txt
resume = docx2txt.process("DAVID MOORE.docx")
text_resume = str(resume)
#new = textFromPdf('new.pdf')



# %%
text_resume = str(resume)
#Summarize the text with ratio 0.1 (10% of the total words.)


# %%
#print(summarize(text_resume, ratio=0.2))
print(summarize(text_resume, ratio=0.2))


# %%
with open('identity.txt') as job:
    text = job.readlines() 

jobContent = ""
for t in text:
    jobContent = jobContent + t.lower().replace("'",'')


# %%
summarize(jobContent, ratio=0.2)


# %%
text_list = [text_resume, jobContent]
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
count_matrix = cv.fit_transform(text_list)


# %%
from sklearn.metrics.pairwise import cosine_similarity


# %%
matchPercentage = cosine_similarity(count_matrix)[0][1] * 100


# %%
matchPercentage = round(matchPercentage, 2)


# %%
print("Your resume matches about "+ str(matchPercentage)+ "% of the job description.")


# %%
terms = keywords(jobContent, ratio=0.25).split('\n')
# Only run nlp.make_doc to speed things up
patterns = [Spnlp.make_doc(t) for t in terms]
matcher.add("Spec", patterns)


# %%
doc = Spnlp(text_resume)
matchkeywords = []
matches = matcher(doc)
for match_id, start, end in matches:
    span = doc[start:end]
    if len(span.text) > 3:
        matchkeywords.append(span.text)


# %%
a = Counter(matchkeywords)
print(a)


# %%
data = []
for t in terms:
    rec={}
    rec['term'] = t
    try:
        rec['influence'] = a[t]
    except:
        rec['influence'] = 0

    data.append(rec)        


# %%
df = pd.DataFrame(data)
df[df['influence'] == 0]['term'].values


# %%
df[df['influence'] > 0]


# %%



