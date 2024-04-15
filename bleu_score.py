import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def calculate_sentiment_score(text):
    doc = nlp(text)
    sentiment_score = doc.sentiment.polarity
    return sentiment_score

def calculate_subjectivity_score(text):
    doc = nlp(text)
    subjectivity_score = doc.sentiment.subjectivity
    return subjectivity_score

news_article_text = """
... (provided example news article text)...
"""

sentiment_score = calculate_sentiment_score(news_article_text)
subjectivity_score = calculate_subjectivity_score(news_article_text)

print(news_article_text)
print(f"\nSentiment Score (spaCy): {sentiment_score:.4f}")
print(f"Subjectivity Score (spaCy): {subjectivity_score:.4f}")