import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import average

def load_models():
    spacy_model_en = spacy.load('en_core_web_lg')
    spacy_model_gr = spacy.load('el_core_news_lg')
    sent_transorfmers_model = SentenceTransformer(
        model_name_or_path = 'sentence-transformers/all-mpnet-base-v2',
        device = 'cpu'
    )
    return [spacy_model_en, spacy_model_gr, sent_transorfmers_model]


def useful_token(token):
    """
    Keep useful tokens which have 
       - Part Of Speech tag (POS): ['NOUN','PROPN','ADJ']
       - Alpha(token is word): True
       - Stop words(is, the, at, ...): False
    """
    return token.pos_ in ['NOUN','PROPN','ADJ'] and token.is_alpha and not token.is_stop and token.has_vector 


def spacy_model_embeddings(corpus, spacy_model):
    doc_vectors = []
    for text_document in corpus:
        doc = spacy_model(text_document)
        vector_list = [token.vector for token in doc if useful_token(token)]
        doc_vector = average(vector_list,axis=0)
        doc_vectors.append(doc_vector)

    return doc_vectors

def sent_transformers_model_embeddings(corpus, sent_transorfmers_model):
    doc_vectors = []
    for text_document in corpus:
        # To modelo tou sent_transorfmers exei periorismo sto length tou text document
        # sent_transorfmers_model.max_seq_length
        # exw grapsei erwthsh gia ton Niko
        doc_vectors.append(sent_transorfmers_model.encode(text_document))
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
 
'''

# ------------------------ ENGLISH DATASET ------------------------ #

# newsgroup website http://qwone.com/~jason/20Newsgroups/

# The sklearn.datasets.fetch_20newsgroups function is a data fetching / caching functions that 
# downloads the data archive from the original 20 newsgroups website, extracts the archive contents in the ~/scikit_learn_data/20news_home folder and calls the sklearn.datasets.load_files on either the training or testing set folder, or both of them:

# https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset

from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(
	subset = 'all', # "train" for the training set, "test" for the test set, "all" for both, with shuffled ordering.
	categories = None, # If None (default), load all the categories, otherwise load categories from scratch(list)
	shuffle = True, # shuffle data
	random_state = 42, # static shuffling->int, total random->None
	remove = (), # ('headers', 'footers', 'quotes') "headers" removes newsgroup headers, "footers" removes blocks at the ends of posts that look like signatures, and "quotes" removes lines that appear to be quoting another post.
	download_if_missing = True, # If False, raise an IOError if the data is not locally available instead of trying to download the data from the source site.
	return_X_y = False # True, returns (data.data, data.target) instead of the total object
)

true_labels = newsgroups_train.target 
print(f"true labels: {newsgroups_train.target}")

X = newsgroups_train.target.data
print(f"data: {newsgroups_train.target.data}")

print(f"file folder: {newsgroups_train.target.filenames}")

print(f"full description of the dataset: {newsgroups_train.target.DESCR}")

feature_names = newsgroups_train.target.target_names
print(f"the names of target classes: {newsgroups_train.target.target_names}")

#print(f"A tuple of two ndarray if return_X_y = True: {newsgroups_train}")




# ------------------------ GREEK DATASET ------------------------ #

from datasets import load_dataset

dataset = load_dataset("greek_legal_code")

'''