import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import average


def useful_token(token):
    """
    Keep useful tokens which have 
       - Part Of Speech tag (POS): ['NOUN','PROPN','ADJ']
       - Alpha(token is word): True
       - Stop words(is, the, at, ...): False
    """
    return token.pos_ in ['NOUN','PROPN','ADJ'] and token.is_alpha and not token.is_stop and token.has_vector 


def spacy_model(corpus, lang):
    # Choose proper model according to dataset
    if (lang == "en"):
        nlp = spacy.load('en_core_web_lg')
    else:
        nlpGr = spacy.load('el_core_news_lg')

    doc_vectors = []
    for text in corpus:
        doc = nlp(text)
        vector_list = [token.vector for token in doc if useful_token(token)]
        doc_vector = average(vector_list,axis=0)
        doc_vectors.append(doc_vector)

    return doc_vectors


def tfidf(corpus):

    vectorizer = TfidfVectorizer(
        lowercase = True,
        use_idf = True,
        norm = None,
        stop_words = "english"
    )

    vectorizer_fitted = vectorizer.fit_transform(corpus)

    feature_names = vectorizer.get_feature_names_out()
    doc_vectors = vectorizer_fitted.todense()

    return [feature_names, doc_vectors]
 



