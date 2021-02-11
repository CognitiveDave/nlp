# -*- coding: utf-8 -*-
"""
Created on Sat May 20 09:51:51 2017

@author: david
https://textblob.readthedocs.io/en/dev/
"""
from textblob import TextBlob
from textblob import Word
import re
from nltk.corpus import stopwords
from textstat.textstat import textstat
from nltk.tokenize import WordPunctTokenizer
#import language_check
from nltk.stem import RegexpStemmer
st = RegexpStemmer('ing$|s$|e$|able$', min=4)
from nltk.stem.snowball import EnglishStemmer
snow = EnglishStemmer()
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from rake_nltk import Rake
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')

class nlp:

    
    def __init__(self):
        self.service = 'TextBlob NLP'
        self.pos_dict = {
            'CC' : 'Coordinating conjunction', 'CD' : 'Cardinal number', 'DT' : 'Determiner',
                'EX' : 'Existential there', 'FW' : 'Foreign word', 'IN' : 'Preposition or subordinating conjunction',
                'JJ' : 'Adjective', 'JJR' : 'Adjective, comparative', 'JJS' : 'Adjective, superlative',
                'LS' : 'List item marker', 'MD' : 'Modal', 'NN' : 'Noun, singular or mass', 'NNS' : 'Noun, plural',
                'NNP' : 'Proper noun, singular', 'NNPS' : 'Proper noun, plural', 'PDT' : 'Predeterminer', 'POS' : 'Possessive ending',
                'PRP' : 'Personal pronoun', 'PRP$' : 'Possessive pronoun', 'RB' : 'Adverb',
                'RBR' : 'Adverb, comparative', 'RBS' : 'Adverb, superlative', 'RP' : 'Particle',
                'SYM' : 'Symbol', 'TO' : 'to', 'UH' : 'Interjection', 'VB' : 'Verb, base form',
                'VBD' : 'Verb, past tense', 'VBG' : 'Verb, gerund or present participle', 'VBN' : 'Verb, past participle',
                'VBP' : 'Verb, non-3rd person singular present', 'VBZ' : 'Verb, 3rd person singular present',  'WDT' : 'Wh-determiner',
                'WP' : 'Wh-pronoun', 'WP$' : 'Possessive wh-pronoun',  'WRB' : 'Wh-adverb'
        }
        self.cachedStopWords = stopwords.words("english")
        #self.tool = language_check.LanguageTool('en-GB')
        self.rake = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
        
    def lang_detect(self,text): 
        giventext = TextBlob(text)
        return giventext.detect_language()

    def others(self,token):
        synonyms = []
        antonyms = []
        for syn in wordnet.synsets(token):
            for l in syn.lemmas():
                synonyms.append(l.name())
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())
        return set(synonyms), set(antonyms)



    def lang_trans(self,text, target):
        giventext = TextBlob(text)
        try:
            trans = giventext.translate(to=target)
            return trans
        except:
            return text
        
    def lang_senti(self,text):
        giventext = TextBlob(text)
        senti = giventext.sentiment
        return senti
        
    def lang_spell(self,text):
        giventext = TextBlob(text)
        words = giventext.words
        corrections = []
        for w in words:
            word = Word(w)
            corr = word.spellcheck()
            if len(corr) > 1:
                corrections.append([word, corr])
        return corrections
 
    def lang_word_counts(self,text):
        giventext = TextBlob(text) 
        words = self.tokenize(text)
        unique_words = set(words)        
        word_freq = {}         
        for u in unique_words:
            word_freq[u] = giventext.words.count(u, case_sensitive=True)        
        return word_freq
     
    def lang_pos(self, text):
        giventext = TextBlob(text)
        tags = giventext.tags
        tag_anal = {}
        for tag in tags:
            try:
                tag_anal[self.pos_dict[tag[1]]] += 1
            except:
                tag_anal[self.pos_dict[tag[1]]] = 1
        return tag_anal
        
    def lang_sent(self, text):
        giventext = TextBlob(text)
        sentences = giventext.sentences
        sent = {}
        sent_senti = []
        sent['count'] = len(sentences)
        for sen in sentences:
            sent_senti.append([sen.string, round(sen.sentiment.polarity,2), round(sen.sentiment.subjectivity,2)])
        sent['sentiment'] = sent_senti    
        return sent        
        
        
    def lang_anal(self,text):
        anal = {}
        pol_rep = 0
        anal['lang'] = self.lang_detect(text)
        anal['original'] = text
        if anal['lang'] != 'en':
            anal['english'] = self.lang_trans(text,'en').string
            text = anal['english']
        sent = self.lang_senti(text)
        pol = round(sent.polarity,2)
        anal['tone'] = pol
        if (pol == 0):
            pol_rep = .5
        elif (pol < 0):
            pol_rep = abs(pol) / 2
        else:
            pol_rep = .5 + (pol / 2)
        anal['polarity'] = pol_rep
        sub = round(sent.subjectivity,2)
        if (sub <.01):
            sub = sub + .005
        anal['subjectivity'] = sub
        cor = self.lang_spell(text)
        if len(cor) > 0:
            anal['corrections'] = cor
        anal['frequencies'] = self.lang_word_counts(text)
        anal['pos'] = self.lang_pos(text)
        anal['sentences'] = self.lang_sent(text)
        anal['stats'] = self.stats(text)
        #anal['quality'] = self.lang_check(text)
        anal['phrases'] = self.keywords(text)
        return anal

 
    def tokenize(self, text):
         min_length = 5
         #tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
         words =  map(lambda word: word.lower(), WordPunctTokenizer().tokenize(text))
         #words = list(WhitespaceTokenizer().span_tokenize(text))
         #words = map(lambda word: word.lower(), word_tokenize(text));    
         #words = wordpunct_tokenize(text)
         #word_tokenize = Penn Treebank#
         #words = map(lambda word: word.lower(), tokenizer.tokenize(text));
         #words = list(parser.raw_parse(text,verbose=False))
         #words = map(lambda word: word.lower(), text.split())
         words = [word for word in words if word not in self.cachedStopWords]
             #tokens =(list(map(lambda token: PorterStemmer().stem(token),words)));
             #tokens =(list(map(lambda token: LancasterStemmer().stem(token),words)));
             #tokens =(list(map(lambda token: snow.stem(token),words)));
             #tokens =(list(map(lambda token: st.stem(token),words)));
         tokens =(list(map(lambda token: wnl.lemmatize(token),words)));
             #tokens=words
         p = re.compile('[a-zA-Z]+');
         filter_words = list(filter(lambda token: p.match(token) and len(token)>=min_length, tokens));
         return filter_words     
  
    def stats(self,text):
          test_data = text
          stats = {}
          stats['flesch_reading_ease'] = textstat.flesch_reading_ease(test_data)
          stats['smog'] = textstat.smog_index(test_data)
          stats['flesch kincaid'] = textstat.flesch_kincaid_grade(test_data) 
          stats['coleman Liau'] = textstat.coleman_liau_index(test_data)
          stats['automated'] = textstat.automated_readability_index(test_data)
          stats['dale chall'] = textstat.dale_chall_readability_score(test_data)
          stats['difficult'] = textstat.difficult_words(test_data)
          stats['linsear'] =  textstat.linsear_write_formula(test_data)
          stats['gunning_fog'] = textstat.gunning_fog(test_data)
          stats['standard'] =  textstat.text_standard(test_data)
          stats['charcount'] = textstat.char_count(test_data)
          stats['lexicon count'] = textstat.lexicon_count(test_data)
          stats['syllable count'] = textstat.syllable_count(test_data)
          stats['sentence count'] = textstat.sentence_count(test_data)
          stats['avg sentence length'] = textstat.avg_sentence_length(test_data)                
          stats['avg_syllables_per_word'] = textstat.avg_syllables_per_word(test_data)
          stats['avg_letter_per_word'] = textstat.avg_letter_per_word(test_data)
          stats['avg_sentence_per_word'] = textstat.avg_sentence_per_word(test_data)          
          return stats
  
    def lang_check(self,text):
        matches = self.tool.check(text)
        msgs = []
        new_text = language_check.correct(text, matches)
        for match in matches:
            msgs.append(match.msg)
        return new_text, msgs
          
    def keywords(self, text):
        keyword = {}
        self.rake.extract_keywords_from_text(text)
        keyword['ranked phrases'] = self.rake.get_ranked_phrases_with_scores()         
        return keyword
