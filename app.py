import streamlit as st
import sys
#Install NLP packages
from textblob import TextBlob
import spacy
from transformers import pipeline

@st.cache
def text_analyzer(text):
    nlp=spacy.load("en")
    docs=nlp(text)
    tokens = [token.text for token in docs]
    alldata=[('"Tokens":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docs]
    return alldata

@st.cache
def named_entity_analyzer(text):
    nlp=spacy.load("en")
    docs=nlp(text)
    tokens = [token.text for token in docs]
    entities=[(entity.text , entity.label_) for entity in docs.ents]
    alldata=['"Tokens":{},\n"Entities":{}'.format(tokens,entities)]
    return alldata


def main():
    st.header("NLP WITH CODE")
    st.subheader("NLP Tasks")

    #Tokenization
    if st.checkbox('Show tokens and Lemma'):
        st.subheader("Tokenize your text")
        message=st.text_area("Enter your Text","Type here")
        if st.button("Analyze"):
            nlp_result=text_analyzer(message)
            st.json(nlp_result)
    #NamedEntity
    if st.checkbox('Show Named Entities'):
        st.subheader("Extract Entities")
        message=st.text_area("Enter your Text","Type here")
        if st.button("Extract"):
            nlp_result=named_entity_analyzer(message)
            st.json(nlp_result)
    #Sentiment Analysis
    if st.checkbox('Show Sentiment Analysis'):
        st.subheader("Sentiment of the input text")
        message=st.text_area("Enter your Text","Type here")
        if st.button("Analyze"):
            blob=TextBlob(message)
            result_sentiment=blob.sentiment
            st.success(result_sentiment)
    #TextSummarization
    if st.checkbox('Show Text Summarization'):
        st.subheader("Summarize your text")
        message=st.text_area("Enter your Text","Type here")
        if st.button("Summarize"):
            summarizer = pipeline("summarization")
            result=summarizer(message, max_length=130, min_length=30, do_sample=False)
            st.success(result)
    st.sidebar.subheader("About the App")
    st.sidebar.text("Performs different NLP tasks")
    st.sidebar.info("NLP has disruptive potential in the modern world")

if __name__=="__main__":
    main()