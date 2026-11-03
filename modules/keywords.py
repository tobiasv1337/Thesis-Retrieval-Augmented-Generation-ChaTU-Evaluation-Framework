import pandas as pd
import markdownify
from markdown import markdown
import codecs
import re
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import logging
import ast

from .config import Config
import pickle
from pathlib import Path

logging.basicConfig(format='%(asctime)s;%(levelname)s - %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

nltk.download('stopwords')
from nltk.corpus import stopwords

additional_stopwords = ["com", "http", "https", "pdf", "www", "mehr", "bzw", "den", "der", "die", "sie", "er", "es",
                        "das", "und", "auf", "des", "beim", "haben", "per", "static", "oder", "mit", "ohne",
                        "möchten", "ein", "ihrer", "wenn", "alle", "bis", "von", "werden", "zum", "sollte", "get",
                        "erst", "must", "ums", "falls", "mache", "fall", "denke", "gibt"]

def parse_list_string(s):
    try:
        # Try to parse as literal list
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed]
    except (ValueError, SyntaxError):
        pass
    # Fallback: split by comma
    return [item.strip() for item in s.split(',')]

_shared_keyword_extractor = None


def get_shared_keyword_extractor(load_only: bool = True):
    global _shared_keyword_extractor
    if _shared_keyword_extractor is None:
        _shared_keyword_extractor = AZKeywordExtractor(load=True, load_only=load_only)
    return _shared_keyword_extractor


class AZKeywordExtractor():

    def __init__(self, load=True, threshold=0.1, depth=None, load_only=False):

        self.threshold = threshold if threshold else 0.1
        self.depth = depth if depth else 999999999999999
        self.languages = ["deutsch", "english"]
        self.path = Config().get("az_data_dir")
        self.cache_path = self.path + "cache/"
        self.mapping = pd.read_csv(self.path + "mapping.csv")
        if load:
            logging.info("Loading TfidfVectorizer from cache.")
            self.tfidf = {}
            try:
                for language in self.languages:
                    self.data = None
                    self.tfidf[language] = pickle.load(open(self.cache_path + f"keyword_vectorizer_{language}.pickle", "rb"))
                return
            except Exception as e:
                if load_only:
                    raise FileNotFoundError(
                        "TF-IDF cache is missing or unreadable. "
                        "Please precompute the keyword_vectorizer_*.pickle files."
                    ) from e
                logging.warning(f"Could not load TfidfVectorizer {language} pickle file. {e}")
        logging.info("Creating TfidfVectorizer.")
        self.data = self.load_website_data()
        self.stopwords = set(list(stopwords.words("german")) + list(stopwords.words("english")) + additional_stopwords)
        self.tfidf = self.fit_tfidf()
        logging.info("AZKeywordExtractor loaded.")

    def load_website_data(self, sample=None):
        rows = []
        blacklist = pd.read_csv(self.path + "blacklist.csv")
        logging.info("Start loading website data: {} Files.".format(len(self.mapping)))
        if sample is None: sample = len(self.mapping)
        error_count = 0
        deutsch_success_count = 0
        english_success_count = 0
        for r, row in self.mapping.head(sample).iterrows():
            if row["depth"] > self.depth: continue
            if r%1000 == 1: logging.info("Loading data: {:.0f}%. Success DE: {:.0f}%. Success EN: {:.0f}%. Errors Total {:.0f}%.".format(r*100/len(self.mapping), deutsch_success_count*100/r, english_success_count*100/r, error_count*100/r))
            if any([row.link == link for link in blacklist.link]): continue
            intent = row["intent"]
            try:
                versions = pd.read_csv(f"{self.path}scraper/topics/{intent}/versions.csv").sort_values(by=["time"], ascending=False).iloc[
                    0]
                html = codecs.open(f"{self.path}scraper/topics/{intent}/content.html", 'r').read()
                md = re.sub(r'\n\n+', '\n', markdownify.markdownify(html, heading_style="ATX")).strip()
                if len(md) > 100000: continue
                text = ''.join(BeautifulSoup(markdown(md), features="lxml").find_all(string=True))
                versions.topic = re.sub(r"^\[['\"](.*)['\"]\]$", r"\1", versions.topic)
                topics = [t for t in [versions.topic, versions.title] + parse_list_string(versions.topics) if not pd.isna(t)]
                rows.append([intent, "deutsch", topics, versions["link"], html, md, text, row["depth"]])
                deutsch_success_count += 1
                html = codecs.open(f"{self.path}scraper/topics/{intent}/content_english.html", 'r').read()
                md = re.sub(r'\n\n+', '\n', markdownify.markdownify(html, heading_style="ATX")).strip()
                if len(md) > 100000: continue
                text = ''.join(BeautifulSoup(markdown(md), features="lxml").find_all(string=True))
                import ast
                versions.english_topic = re.sub(r"^\[['\"](.*)['\"]\]$", r"\1", versions.english_topic)
                topics = [t for t in [versions.english_topic, versions.english_title] + parse_list_string(versions.english_topics) if not pd.isna(t)]
                rows.append([intent, "english", topics, versions["mapping"], html, md, text, row["depth"]])
                english_success_count += 1
            except Exception as e:
                #print(versions)
                #print(intent, e)
                error_count += 1
                continue
        return pd.DataFrame(rows, columns=["intent", "language", "topics", "link", "html", "md", "text", "depth"])

    def fit_tfidf(self):
        tfidf = {}
        for language in self.languages:
            logging.info(f"Fitting tfidf for language: {language}")
            tfidf[language] = TfidfVectorizer(analyzer='word', stop_words=self.stopwords)
            tfidf[language].fit(self.data.loc[self.data.language == language, "text"])
            Path(self.cache_path).mkdir(parents=True, exist_ok=True)
            pickle.dump(tfidf[language], open(self.cache_path + f"keyword_vectorizer_{language}.pickle", "wb"))
        return tfidf

    def get_keywords(self, utterance, language="deutsch"):
        values = np.array(self.tfidf[language].transform([utterance]).todense())[0]
        names = np.array(self.tfidf[language].get_feature_names_out())
        names_clean = names[values > self.threshold]
        #print("Names:", names_clean)
        return reconstruct_sentence(utterance, names_clean)

    def get_keywords_batch(self, texts, language="deutsch"):
        values, names = self.get_transform(texts, language)
        keyword_sentences = []
        for (text, value) in zip(texts, values > self.threshold):
            keywords = names[value]
            if len(keywords) < 1:
                keyword_sentences.append("")
            else:
                keyword_sentences.append(reconstruct_sentence(text, keywords))
        return keyword_sentences

    def get_transform(self, texts, language="deutsch"):
        values = np.array(self.tfidf[language].transform(texts).todense())
        names = np.array(self.tfidf[language].get_feature_names_out())
        return values, names

def reconstruct_sentence(utterance, keywords):
    sentence = ""
    for word in utterance.split(" "):
        if re.sub('[^a-zA-ZäöüÄÖÜß\s]+', '', word).lower() not in keywords and not any(
                [name in word.lower() for name in keywords]): continue
        sentence += " " + word
    return sentence.strip()