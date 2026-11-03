#!/usr/bin/python

import logging
import time
import html
import json
import os
import shutil
import requests
import unidecode
import string
import regex as re
import pandas as pd
import numpy as np
from random import randrange
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse

from .translator import translate_api
from .config import Config


SCRAPE_GET_TIMEOUT = (3.05, 27)
SCRAPE_GET_RETRIES = 3
SCRAPE_GET_RETRY_BACKOFF_SECONDS = (1, 2, 4)
SCRAPE_GET_RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _get_with_retries(url, timeout=SCRAPE_GET_TIMEOUT):
    """GET a scrape target while preserving the legacy response semantics."""
    for attempt in range(SCRAPE_GET_RETRIES + 1):
        try:
            response = requests.get(url, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as error:
            if attempt == SCRAPE_GET_RETRIES:
                raise
            delay = SCRAPE_GET_RETRY_BACKOFF_SECONDS[attempt]
            logging.warning(
                f"Retrying GET {url} after {type(error).__name__} "
                f"(attempt {attempt + 2}/{SCRAPE_GET_RETRIES + 1}, "
                f"backoff {delay}s)"
            )
            time.sleep(delay)
            continue

        status_code = response.status_code
        if status_code in SCRAPE_GET_RETRY_STATUS_CODES and attempt < SCRAPE_GET_RETRIES:
            delay = SCRAPE_GET_RETRY_BACKOFF_SECONDS[attempt]
            logging.warning(
                f"Retrying GET {url} after HTTP {status_code} "
                f"(attempt {attempt + 2}/{SCRAPE_GET_RETRIES + 1}, "
                f"backoff {delay}s)"
            )
            time.sleep(delay)
            continue
        if status_code in SCRAPE_GET_RETRY_STATUS_CODES:
            logging.warning(
                f"GET {url} still returned HTTP {status_code} after "
                f"{SCRAPE_GET_RETRIES + 1} attempts; preserving the response"
            )
        elif attempt:
            logging.info(f"GET {url} recovered after {attempt + 1} attempts")
        return response

    raise AssertionError("unreachable")


def scrape_az_website():
    path = Config().get("az_data_dir")#"data/az/"
    logging.info("Start: Run Scraper")
    Scraper(path, "scraper", "topics").run()  # Run the web scraping task
    logging.info("Done: Run Scraper")
    logging.info("Start: Analyze & Postprocess AZ")
    from .analysis import analyze_scraped_data_az
    limit = 10
    while analyze_scraped_data_az(path):
        if limit < 1: break
        logging.info("Start: Found Issue, Rerunning Scraper")
        Scraper(path, "scraper", "topics").run()
        logging.info("Done: Found Issue, Rerunning Scraper")
        limit -= 1
        logging.info("Start: Analyze & Postprocess AZ")
    logging.info("Done: Analyze & Postprocess AZ")
    logging.info("Start: Automatic Translator")
    Translator(path, "scraper", "topics").run()  # Run the web scraping task
    logging.info("Done: Automatic Translator")
    logging.info("Start: Generate Keyword Extractor")
    from .keywords import AZKeywordExtractor
    az_extractor = AZKeywordExtractor(load=False)
    logging.info("Done: Generate Keyword Extractor")
    logging.info("Start: Generate Button Titles")
    from .title_generation import generate_titles_for_mapping
    generate_titles_for_mapping(
        az_extractor,
        path,
        generate_llm_titles=bool(Config().get("generate_llm_titles", True)),
    )
    logging.info("Done: Generate Button Titles")
    logging.info("Start: Generate Faiss Index for RAG")
    from .rag import ChatQA
    ChatQA().load_or_create_faiss(recreate=True)
    logging.info("Done: Generate Faiss Index for RAG")



def replace(text, replace_this, replace_with=""):
    """
    Iterates through each substring in 'replace_this' and replaces with the specified 'replace_with'
    string in the input 'text'.

    Args:
    text (str): The input text where replacements will be made.
    replace_this (list of str): A list of substrings to be replaced.
    replace_with (str, optional): The string to replace 'replace_this' with.
        Defaults to an empty string.

    Returns:
    str: The modified text with replacements.
    """
    for remove in replace_this:
        while remove in text:
            text = text.replace(remove, replace_with)
    return text

def clean_url(url):
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip('/'), '', '', parsed.fragment))

def get_link(link, base_url="https://www.tu.berlin"):
    if not link or len(link) < 2: return link
    link = html.unescape(link)
    parsed = urlparse(link)
    if parsed.scheme in ['http', 'https']: return clean_url(link)
    if link.startswith('/'):  return clean_url(urljoin(base_url, link))
    return link  # skip relative junk like "foo", "#bar", "mailto:"


def decode(text):
    return unidecode.unidecode(
        text.lower().translate(
            str.maketrans('', '', string.punctuation)
        )
    ).replace(" ", "-")\
        .replace(",","")\
        .replace('"',"")\
        .replace("'","")


def check_anchor_exists(url, anchor, html=None):
    """
    Check if an anchor (by id or name) exists on a page.

    Args:
        url (str): The URL of the page to check
        anchor (str): The anchor name/id to look for
        html (BeautifulSoup, optional): Pre-fetched HTML to use instead of fetching

    Returns:
        bool: True if anchor exists, False otherwise
    """
    if not anchor or not url:
        return False

    try:
        if html is None:
            r = _get_with_retries(url)
            html = BeautifulSoup(r.text.strip(), 'html.parser')

        # Check for element with matching id
        if html.find("div", {"id": anchor}) is not None:
            return True
        # Check for element with matching name attribute
        if html.find(attrs={"name": anchor}) is not None:
            return True
        # Check for any element with matching id
        if html.find(id=anchor) is not None:
            return True

        return False
    except Exception as e:
        logging.warning(f"check_anchor_exists - {url} #{anchor}: {e}")
        return False


class Translator:
    """
    Class for translator.

    Attributes:
    topics_path (str): The path to directories containing topic data.
    subdirectories (list): List of subdirectories within 'topics_path'.
    """

    def __init__(self, path, scraper_folder, topics_folder):
        """
        Initialize the Translator object.

        Args:
        topics_path (str): The path to directories containing all topic folders.
        """
        self.path = path
        self.topics_path = os.path.join(self.path, scraper_folder, topics_folder)
        self.subdirectories = os.listdir(self.topics_path)

    def read_and_save(self):
        """
        Read version files in subdirectories and save translated data.

        Iterates through subdirectories, reads version files, translates text,
        and saves the translated data to the original files.
        """
        for subdir in self.subdirectories:
            subdir_path = os.path.join(self.topics_path, subdir)
            if os.path.isdir(subdir_path):
                version_file_path = os.path.join(subdir_path, 'versions.csv')
                version = pd.read_csv(version_file_path)
                version = version.replace(np.nan, '', regex=True)

                if version["english_topics"].iloc[-1] == '':
                    logging.info(f"translate topics: {subdir_path}")
                    version = self.translate_column(version, "topics")
                if version["english_topic"].iloc[-1] == '':
                    logging.info(f"translate topic: {subdir_path}")
                    version = self.translate_column(version, "topic")
                if version["english_title"].iloc[-1] == '' and version["title"].iloc[-1] != '':
                    logging.info(f"translate title: {subdir_path}")
                    version = self.translate_column(version, "title")
                version.to_csv(version_file_path, index=False)

    def translate_column(self, df, column):
        """
        Translate the content of a DataFrame column from german to english using the GoogleTranslator API.

        Args:
        df (pandas.DataFrame): DataFrame with all data.
        column (str): column to translate.

        Returns:
        pandas.DataFrame: Updated DataFrame with translated content.
        """
        translated_column = "english_" + column
        last_row = df.iloc[-1]
        text = str(last_row[column])[:5000]  # translation API allows max. 5000 character
        df.loc[df.index[-1], translated_column] = translate_api(utterances=[text], cache_path=self.path, source='de', target='en')
        return df

    def run(self):
        self.read_and_save()


class Scraper:
    """
    A class for web scraping tasks.

    Initializes with additional topics and performs web scraping using the AZ library.

    Attributes:
    az_deutsch (az.AZ): An instance of the AZ class for web scraping.
    """

    def __init__(self, path, scraper_folder, topics_folder):
        """
        Initialize the Scraper object.

        Additional topics for web scraping are provided for scraping.

        Attributes:
        az_deutsch (az.AZ): An instance of the AZ class for web scraping.
        """

        self.path = path
        self.topics_folder = os.path.join(self.path, scraper_folder, topics_folder)
        self.mapping_file = os.path.join(self.path, 'mapping.csv')

        additional_topics = pd.DataFrame([
            ["studierendensekretariat", "Studierendensekretariat", "https://www.tu.berlin/studieren/studienorganisation/themen-a-z/"],
            ["bewerben-und-einschreiben", "Bewerben und Einschreiben",
             "https://www.tu.berlin/studieren/bewerben-und-einschreiben"],
            ["pruefungen", "Prüfungen", "https://www.tu.berlin/studieren/studienorganisation/pruefungen/"],
            ["studienorganisation", "Studienorganisation", "https://www.tu.berlin/studieren/studienorganisation"],
            # ["studieren", "Studieren", "https://www.tu.berlin/studieren"],
            # ["main", "Hauptseite", "https://www.tu.berlin"],
            ["lernraeume", "Lehrräume", "https://www.tu.berlin/studieren/uni-leben/lernraeume"],
            ["it-dienste", "IT-Dienste", "https://www.tu.berlin/studieren/studienorganisation/it-dienste"],
            ["campus-leben", "Campus-Leben", "https://www.tu.berlin/themen/campus-leben"],
            ["studienstart", "Studienstart", "https://www.tu.berlin/international/starten-an-der-tu-berlin"],
            ["studierendenservice", "Studierendenservice", "https://www.tu.berlin/studierendenservice"],
            ["studyguide", "Studyguide", "https://www.tu.berlin/studienberatung/studyguide"],
            ["studienangebot", "Studienangebot", "https://www.tu.berlin/studieren/studienangebot/gesamtes-studienangebot"]
        ], columns=["intent", "topic", "link"])

        self.az_deutsch = AZ(
            path=self.path,
            topics_path=self.topics_folder,
            link="https://www.tu.berlin/studieren/studienorganisation/themen-a-z/",
            additional_topics=additional_topics,
            language="Deutsch"
        )

    def delete_tmp_topic_folder(self):
        if os.path.exists(self.topics_folder + ".tmp"):
            shutil.rmtree(self.topics_folder + ".tmp")
        if os.path.exists(self.mapping_file + ".tmp"):
            os.remove(self.mapping_file + ".tmp")

    def delete_old_topic_folder(self):
        if os.path.exists(self.topics_folder):
            shutil.rmtree(self.topics_folder)
        if os.path.exists(self.topics_folder + ".tmp"):
            os.rename(self.topics_folder + ".tmp", self.topics_folder)
        if os.path.exists(self.mapping_file):
            os.remove(self.mapping_file)
        if os.path.exists(self.mapping_file + ".tmp"):
            os.rename(self.mapping_file + ".tmp", self.mapping_file)

    def run(self):

        logging.info("Start: Delete Old Tmp Topic Folders")
        self.delete_tmp_topic_folder()
        logging.info("Done: Delete Old Tmp Topic Folders")

        self.az_deutsch.run()

        logging.info("Start: Delete Old Topic Folders")
        self.delete_old_topic_folder()
        logging.info("Done: Delete Old Topic Folders")


class AZ:
    """
    A class for web scraping and topic mapping.

    Initializes with the necessary parameters for web scraping and topic mapping.
    Provides methods for scraping web content and managing topics.

    Attributes:
    path (str): The path where scraped data will be saved.
    link (str): The base URL for web scraping.
    additional_topics (pd.DataFrame): Additional topics to scrape.
    save_row (bool): Flag to control whether to save scraped rows.
    topics (dict): A dictionary to store topic objects.
    intents (list): A list to store topic intents.
    mapped (AZ): A mapping instance of AZ (German-English, English-German).
    queue (list): A queue to manage topic scraping.
    """
    def __init__(self, path, topics_path, link=None, additional_topics=None, save_row=True, language="Deutsch"):
        """
        Initialize the AZ object.

        Args:
        path (str): The path where scraped data will be saved.
        link (str, optional): The base URL for web scraping. Defaults to None.
        additional_topics (pd.DataFrame, optional): Additional topics to scrape. Defaults to None.
        save_row (bool, optional): Flag to control whether to save scraped rows. Defaults to True.
        """
        self.blacklist = pd.read_csv(os.path.join(path, "blacklist.csv"))["link"].unique()

        self.path = path
        self.old_topics_path = topics_path
        self.topics_path = topics_path + ".tmp"
        self.mapping_file = os.path.join(self.path, "mapping.csv.tmp")
        self.old_mapping_file = os.path.join(self.path, "mapping.csv")
        self.link = link if link[-1] != "/" else link[:-1]
        self.additional_topics = additional_topics
        self.language = language

        self.topics = {}
        self.intents = []
        self.mapped = None
        self.save_row = save_row
        self.queue = []

    def run(self):
        """
        Run the web scraping and topic mapping process.
        """
        self.reset()
        self.save_row = True
        logging.info(f"# Final Scrape")
        self.scrape_all()

    def reset(self):
        """
        Reset the topic and intent lists.
        """
        self.topics = {}
        self.intents = []
        self.queue = []

    def scrape_all(self, queue=True):
        """
        Scrape all topics.

        Args:
        queue (bool, optional): Flag to control whether to use a queue for scraping. Defaults to True.
        """
        self.link = self.add_topic(self.link, None, "AZ")   # add first topic of AZ
        if self.additional_topics is not None:  # add all additional topics
            for t, topic in self.additional_topics.iterrows(): self.add_topic(topic.link, None, topic.topic)
        self.topics[self.link].finish()  # add all subpages
        if not queue: return
        finish_count = 0
        while len(self.queue) > 0:  # process all topics in queue
            logging.info(f"Finish {self.queue[0].link} (depth {self.queue[0].depth}) - {finish_count} finished, {len(self.queue)} in queue")
            self.queue.pop(0).finish()  # process the next one
            finish_count += 1

    def add_topic(self, link, parent, topic, intent=None, save=None, queue=True):
        """
        Add a topic in queue from scraping.

        Args:
        link (str): The URL of the topic.
        parent (str, list): The parent topic(s).
        topic (str): The topic name.
        intent (str, optional): The intent associated with the topic. Defaults to None.
        """
        save = self.save_row if save is None else save
        link = link if link[-1] != "/" else link[:-1]
        new_topic = None
        if link not in self.topics:
            logging.info(f'Add {"" if self.save_row else "EN"} {"" if topic is None else topic} ({link}) {"" if parent is None else "from parent (" + parent + ")"}')
            parents = [] if parent is None else [parent]
            if topic is None:
                new_topic = Topic(self, link, parents, save=save, intent=intent)
            else:
                new_topic = Topic(self, link, parents, [topic], save=save, intent=intent)
            link = new_topic.link
        if link in self.topics:
            self.topics[link].additional_topic(topic)
            self.topics[link].additional_parent(parent)
        elif new_topic is not None:
            if queue: self.queue.append(new_topic)
            self.topics[link] = new_topic
        return link

    def get_mapped_link_of_topic(self, link):
        """
        Get the mapped link of a link.

        Args:
        link (str): The URL of a topic.

        Returns:
        str: The mapped link (german-english, english-german) of the topic.
        """
        if link[-1] == "/": link = link[:-1]
        return self.topics[link].mapping

    def get_mapped_topic(self, link):
        """
        Get the mapped topic of a link.

        Args:
        link (str): The URL of a topic.

        Returns:
        Topic: The mapped topic object.
        """
        if link[-1] == "/": link = link[:-1]
        mapped_link = self.get_mapped_link_of_topic(link)
        if mapped_link not in self.mapped.topics: return None
        return self.mapped.topics[mapped_link]

    def add_mapped_topic(self, link, parent, topic, intent=None):
        """
        Add a mapped topic in the mapped AZ instance.

        Args:
        link (str): The URL of an original topic.
        parent (str or list): The parent topic(s) of the mapped topic.
        topic (str): The name of the mapped topic.
        intent (str, optional): The intent associated with the mapped topic. Defaults to None.
        """
        if self.mapped is None:
            self.mapped = AZ(
                path=self.path,
                topics_path=self.topics_path,
                link=self.get_mapped_link_of_topic(self.link),
                save_row=False,
                language="Deutsch" if self.language == "English" else "Deutsch"
            )
        return self.mapped.add_topic(link, parent, topic, intent=intent)

    def scrape_mapped_topics(self):
        self.mapped = AZ(
            path=self.path,
            topics_path=self.topics_path,
            link=self.get_mapped_link_of_topic(self.link),
            additional_topics=None if self.additional_topics is None else pd.DataFrame(
                [[None, None, self.get_mapped_link_of_topic(link)] for link in self.additional_topics.link if
                 self.get_mapped_link_of_topic(link) is not None], columns=["intent", "topic", "link"]),
            save_row=False,
            language="English" if self.language == "Deutsch" else "English"
        )
        self.mapped.scrape_all()


class Topic:

    def __init__(self, az, link, parents=None, topics=None, save=True, anchor=None, html=None, content=None, mapping=None, name=None, intent=None, topic=None, etag=None):

        self.az = az
        self.link = link# if link[-1] != "/" else link[:-1]
        self.parents = parents if parents is not None else []
        self.depth, self.parent = self.update_depth()


        self.name = name
        self.html = html
        self.content = content
        self.mapping = mapping
        self.topic = topic
        self.title = self.topic
        self.topics = topics if topics is not None else []
        self.intent = intent
        self.old_intents = []
        self.anchor = anchor

        self.etag = etag
        self.old_intent_name = None

        self.save = save

        self.english_topic = None
        self.english_title = None
        self.english_topics = []
        self.english_html = None
        self.english_content = None

        self.sub_pages = []

        self.fill_missing_data()

    def get_old_intent_name(self):
        try:
            link_intent_mapping = pd.read_csv(self.az.old_mapping_file).drop_duplicates(subset="link", keep="first").set_index("link").to_dict(
                "index")
            self.etag = self.etag if self.etag is not None or self.link not in list(link_intent_mapping) else \
            link_intent_mapping[self.link]["etag"]
            if self.link in link_intent_mapping:
                self.old_intent_name = link_intent_mapping[self.link]["intent"]
        except FileNotFoundError:
            # logging.warning(f"no saved file for {self.link}")
            pass

    def fill_missing_data(self):
        self.get_link()
        if self.html is None: self.get_html()
        if self.topic not in self.topics: self.additional_topic(self.topic)
        if self.content is None: self.get_content()
        if self.intent is None:
            if self.topics is None or len(self.topics) < 1 or "#" not in self.link: self.get_topics()
            self.choose_topic()
        self.set_intent()
        if self.mapping is None or self.english_html is None: self.get_mapping()

    def finish(self):
        if self.link not in self.az.blacklist and self.link.split("/")[-1] not in ["aktuelles", "news"]:
            if len(self.sub_pages) < 1: self.get_subpages()
            else:
                self.sub_pages = [self.az.add_topic(sub_page_link, self.link, None) for sub_page_link in self.sub_pages]
        else:
            self.sub_pages = []
        self.save_link_intent_mapping()
        self.save_topic()

    def topic_file(self, intent, file=None):
        if file is None:
            return os.path.join(self.az.topics_path, intent)
        return os.path.join(self.az.topics_path, intent, file)

    def old_topic_file(self, intent, file=None):
        if file is None:
            return os.path.join(self.az.topics_path, intent)
        return os.path.join(self.az.topics_path, intent, file)

    def save_topic(self):
        if not self.save or not self.az.save_row: return
        path = self.topic_file(self.intent)
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            logging.warning(f"save_topic - FileExistsError?? {path}")
            os.remove(path)
            Path(path).mkdir(parents=True, exist_ok=True)
        try:
            df = pd.read_csv(path+"/versions.csv")
            if len(df) < 1: raise FileNotFoundError
            header = False
        except FileNotFoundError:
            header = True

        with open(path+"/page.html", "w", encoding='utf-8') as file:
            file.write(str(self.html))
        with open(path+"/content.html", "w", encoding='utf-8') as file:
            file.write(str(self.content))
        if self.mapping is not None:
            try:
                if self.english_html is not None:
                    with open(path+"/page_english.html", "w", encoding='utf-8') as file:
                        file.write(str(self.english_html))
                if self.english_content is not None:
                    with open(path+"/content_english.html", "w", encoding='utf-8') as file:
                        file.write(str(self.english_content))
            except Exception as e:
                logging.warning(f"save_topic - {self.link} {e}")

        row, columns = self.get_row()
        columns = ["time"]+columns
        if os.path.isfile(path+"/versions.csv"):
            try:
                pd.read_csv(path+"/versions.csv")[columns].to_csv(path+"/versions.csv", index=False)
            except Exception as e:
                logging.warning(f"save_topic - {e}")
                print(pd.read_csv(path+"/versions.csv"))
                pd.DataFrame([[]], columns=columns).to_csv(path+"/versions.csv", index=False)
        pd.DataFrame([[time.time()]+row], columns=columns).to_csv(path+"/versions.csv", mode='a', header=header, index=False)

    def save_link_intent_mapping(self):
        if not self.save: return
        mapping_columns = ["depth", "extracted_title_deutsch", "extracted_title_english", "intent", "link", "mapping",
                           "parent", "etag"]
        mapping_values = [self.depth, self.topic, self.english_topic, self.intent, self.link, self.mapping, self.parent,
                          self.etag]
        title_columns = ["generated_title_deutsch", "generated_title_english", "manual_title_deutsch",
                         "manual_title_english"]

        # Load old mapping.csv to preserve title columns. On a fresh setup this
        # file does not exist yet, so start with an empty mapping.
        try:
            old_mapping = pd.read_csv(self.az.old_mapping_file)
        except FileNotFoundError:
            old_mapping = pd.DataFrame(columns=mapping_columns + title_columns)

        # Load current (new) mapping being built
        try:
            mapping = pd.read_csv(self.az.mapping_file)
        except FileNotFoundError:
            mapping = pd.DataFrame(columns=mapping_columns)

        # Ensure title columns exist in mapping
        for col in title_columns:
            if col not in mapping.columns:
                mapping[col] = None

        # Step 1: Check if link already exists in current mapping (safety guard for unexpected duplicates)
        if self.link in list(mapping.link):
            logging.warning(f"Intent Already Exists! {self.intent} {self.link}")
            logging.warning(mapping[mapping.link == self.link].values)
            # mapping.loc[mapping.link == self.link, mapping_columns] = mapping_values
            # what needs to happen when this occurs?

        # Create new row and preserve title columns from old mapping
        new_row = pd.DataFrame([mapping_values], columns=mapping_columns)
        old_row = old_mapping[old_mapping.link == self.link].iloc[0] if self.link in list(old_mapping.link) else None
        for col in title_columns:
            new_row[col] = old_row[col] if old_row is not None and col in old_mapping.columns and pd.notna(
                old_row[col]) else None

        mapping = pd.concat([mapping, new_row], ignore_index=True)
        mapping.drop_duplicates(subset=mapping_columns, keep="first").to_csv(self.az.mapping_file, index=False)

    def get_row(self, columns=None, return_columns=True):
        if columns is None:
            columns = ["depth", "intent", "old_intents", "topic", "english_topic", "title", "english_title",
                       "topics", "english_topics", "parent", "parents", "name", "sub_pages", "link", "anchor", "mapping", "etag"]
        row = [
            self.depth,
            self.intent,
            ", ".join(self.old_intents),
            self.topic,
            self.english_topic,
            self.title,
            self.english_title,
            ", ".join(self.topics),
            ", ".join(self.english_topics),
            self.parent,
            ", ".join(self.parents),
            self.name,
            ", ".join(self.sub_pages),
            self.link,
            self.anchor,
            self.mapping,
            self.etag
        ]
        row = [re.sub(' +', ' ', str(r).replace("\n","\\n")) if r is not None else r for r in row]
        if return_columns: return row, columns
        else: return row


    def get_link(self):
        self.link = get_link(self.link)
        if "#" in self.link: self.anchor = self.link.split("#")[-1]
        #if self.link[-1] == "/": self.link = self.link[:-1]

    def get_topics(self):
        link = self.link.replace("https://","").replace("http://","").replace("www.","")
        link = re.sub('([A-Za-z]+\.html+)','', link)
        link = re.sub('([A-Za-z]+\.php+)','', link)
        link = re.sub('(\.[a-z]+)','', link)
        link_split = link.split("?")[0].split("/")
        #while "." in link_split[-1]: link_split.pop(-1)
        name = link_split[-1].split("#")[0] if len(link_split[-1].split("#")[0]) > 1 else link_split[-2].split("#")[0]
        if re.sub(r'[^a-z]+', '', name.lower()).strip() in ["go"]: name = ""
        if self.topics is None: self.topics = []
        if len(name) > 2: self.topics.append(name.replace("-", " ").title())

    def choose_topic(self):
        #self.topic = self.topics[0]
        if isinstance(self.name, str) and self.name not in self.topics: self.topics.append(self.name.title())
        if isinstance(self.title, str) and self.title not in self.topics: self.topics.append(self.title.title())
        if isinstance(self.topic, str) and self.topic not in self.topics: self.topics.append(self.topic.title())
        if len(self.topics) < 1:
            link_topics = re.findall("[\/]([a-zA-Z\-]+)[\/\#]", self.link)
            if len(link_topics) > 0: self.topics.append(link_topics[-1].title())
            else:
                link_topics = re.findall("[\.]([a-zA-Z\-]+)[\.]", self.link)
                if len(link_topics) > 0: self.topics.append(link_topics[-1].title())
                else:
                    link = self.link.replace("https://", "").replace("http://", "").replace("www.", "")
                    link = re.sub('(\.[a-z]+)', '', link)
                    self.topics.append(link.title())
        self.topic = min(self.topics, key=len)

    def get_name(self):
        self.name = re.sub(' +', ' ', decode(self.topic.lower().translate(str.maketrans('', '', string.punctuation))).strip())\
            .replace("\n", "")\
            .replace(" ", "-")

        topics = self.topics.copy()
        while self.topic in topics and len(topics) > 1 and any([self.name == self.az.topics[parent].name for parent in self.parents]):
            topics.remove(self.topic)
            self.topic = min(topics, key=len)
            self.name = re.sub(' +', ' ', decode(
                self.topic.lower().translate(str.maketrans('', '', string.punctuation))).strip()).replace("\n"," ")\
                .replace(" ", "-")

    def set_intent(self):
        if self.intent is None:
            if self.name is None: self.get_name()
            if self.parents is None or len(self.parents) < 1: self.intent = self.name
            else: self.intent = self.name + "-" + self.az.topics[self.parents[0]].name
        i, new_intent = 2, self.intent
        while new_intent in self.az.intents:
            new_intent = self.intent + str(i)
            i += 1
        whitelist = set(' -_abcdefghijklmnopqrstuvwxyz1234567890')
        self.intent = ''.join(filter(whitelist.__contains__, new_intent.lower()))
        self.az.intents.append(new_intent)

    def get_canonical_link(self):
        # returns true if the link changed
        canonical_link = self.html.find("link", {"rel": "canonical"})
        if canonical_link is not None:
            canonical_link = canonical_link["href"]
            if self.anchor is not None: canonical_link = f"{canonical_link}#{self.anchor}"
            if self.link != canonical_link and len(canonical_link.strip()) > 0:
                logging.info(f"Canonical link retrieved. Changing {self.link} to {canonical_link}")
                self.link = canonical_link
                self.get_old_intent_name()
                return True
        return False

    def get_html(self):
        if ("tu-berlin" not in self.link and "tu.berlin" not in self.link) or any([f".{ending}" in self.link.lower() for ending in ["pdf","png","jpg","mp4","mp3"]]) or any([self.link.startswith(link) for link in [" https://tu-berlin.zoom.us", "https://ipac.ub.tu-berlin.de/ipactu/catalog/toc", "https://portal.ub.tu-berlin.de", "https://www.summer-university.tu-berlin.de", "https://portal.ub.tu-berlin.de", "https://hpc.tu-berlin.de"]]):
            self.etag = randrange(1000000000)
            return
        try:
            r = _get_with_retries(self.link)
            etag = r.headers["Etag"].replace('"',"") if "Etag" in r.headers else randrange(1000000000)
        except Exception as e:
            logging.warning(f"get_html header - {self.link}: {e}")
            return
        if self.save: self.get_old_intent_name()
        reuse = False #
        if self.etag is None or etag != self.etag or self.old_intent_name is None or not os.path.isfile(self.old_topic_file(self.old_intent_name, "versions.csv")) or etag not in list(pd.read_csv(self.old_topic_file(self.old_intent_name, "versions.csv")).etag):
            try:
                self.html = BeautifulSoup(r.text.strip(), 'html.parser')
                if self.get_canonical_link() and etag == self.etag and os.path.isfile(self.old_topic_file(self.old_intent_name, "versions.csv")) and etag in list(pd.read_csv(self.old_topic_file(self.old_intent_name, "versions.csv")).etag):
                    reuse = True
            except Exception as e:
                logging.warning(f"get_html page - {self.link}: {e}")
            self.etag = etag
        else:
            reuse = True
        if reuse:
            df = pd.read_csv(self.old_topic_file(self.old_intent_name, "versions.csv")).fillna(np.nan).replace([np.nan], [None])
            df = df[df.etag == etag].iloc[0]# if etag in df.etag else df = df.iloc[-1]
            self.mapping = df["mapping"]
            self.anchor = df["anchor"]
            self.sub_pages = df["sub_pages"].split(", ") if isinstance(df["sub_pages"], str) else []
            self.old_intents = df["old_intents"].split(", ") if isinstance(df["old_intents"], str) else []
            if df["intent"] not in self.old_intents: self.old_intents.append(df["intent"])
            self.title = df["title"]
            for topic in str(df["topics"]).split(", "): self.additional_topic(topic)
            self.english_topic = df["english_topic"]
            self.english_title = df["english_title"]
            self.english_topics = df["english_topics"].split(", ") if isinstance(df["english_topics"], str) else []
            if not isinstance(self.english_topic, str) and isinstance(df["english_topics"], str): self.english_topic = min(self.english_topics)
            fname = self.old_topic_file(self.old_intent_name, "page.html")
            if os.path.isfile(fname):
                with open(fname) as fp:
                    self.html = BeautifulSoup(fp, 'html.parser')
            else:
                self.html = BeautifulSoup(_get_with_retries(self.link).text.strip(), 'html.parser')
            fname = self.old_topic_file(self.old_intent_name, "content.html")
            if os.path.isfile(fname):
                with open(fname) as fp:
                    self.content = BeautifulSoup(fp, 'html.parser')
            fname = self.old_topic_file(self.old_intent_name, "page_english.html")
            if os.path.isfile(fname):
                with open(fname) as fp:
                    self.english_html = BeautifulSoup(fp, 'html.parser')
            elif not pd.isna(self.mapping):
                self.english_html = BeautifulSoup(_get_with_retries(self.mapping).text.strip(), 'html.parser')
            fname = self.old_topic_file(self.old_intent_name, "content_english.html")
            if os.path.isfile(fname):
                with open(fname) as fp:
                    self.english_content = BeautifulSoup(fp, 'html.parser')

    def check_language(self, language_item):
        if not language_item.has_attr('title'): return
        if language_item['title'] not in ['Deutsch', 'English']: return
        if language_item['title'] != self.az.language: return
        new_topic = Topic(self.az, self.mapping, self.parents, save=self.az.save_row)
        logging.info(f"Switching {self.link} to {self.az.language} version: {self.mapping}")
        self.__dict__.clear()
        self.__dict__.update(new_topic.__dict__)

    def get_mapping(self):
        if self.html is None: return
        if self.az.save_row is False: return
        if not isinstance(self.mapping, str):
            header = self.html.find("header")
            if header is not None:
                language_menu_items = header.find_all("div", {"class": "languageMenu__item"})
                for language_menu_item in language_menu_items:
                    if "languageMenu__item--active" in language_menu_item["class"]: continue
                    language_menu_item_link = language_menu_item.find("a")
                    if language_menu_item_link is None: continue
                    if not language_menu_item_link.has_attr('href') or len(language_menu_item_link["href"]) < 1: continue
                    self.mapping = get_link(language_menu_item_link["href"])
                    self.check_language(language_menu_item_link)
                    break
            if self.anchor is not None:
                anchor_html = self.html.find("div", {"id": self.anchor})
                if anchor_html is not None: self.content = anchor_html
                if self.mapping is not None:
                    if check_anchor_exists(self.mapping, self.anchor, html=self.english_html):
                        self.mapping = f"{self.mapping}#{self.anchor}"
                    else:
                        self.mapping = None
        if not isinstance(self.mapping, str): return
        if self.english_html is not None: return
        self.save = True
        try:
            intent = self.old_intent_name if self.old_intent_name is not None else self.intent
            self.mapping = self.az.add_mapped_topic(self.mapping, None, None, intent=intent)
            mapped_topic = self.az.mapped.topics[self.mapping]
            self.english_topic = mapped_topic.topic
            self.english_title = mapped_topic.title
            self.english_topics = mapped_topic.topics
            self.english_html = mapped_topic.html
            self.english_content = mapped_topic.content
        except Exception as e:
            logging.warning(f"get_mapping - {self.link} {self.mapping}: {e}")

    def additional_topic(self, topic):
        if not isinstance(topic, str) or topic == "nan" or len(re.findall(r'[a-zA-Z]', topic)) < 3: return
        topic = topic.replace(",", "٬")
        if len(self.topics) < 1: self.topics = [topic]
        else: self.topics.append(topic)

    def additional_parent(self, parent):
        if len(self.parents) < 1: self.parents = [parent]
        else: self.parents.append(parent)
        self.update_depth()

    def update_depth(self):
        # ensure all parents are "real"
        self.parents = [parent for parent in self.parents if parent in self.az.topics]
        if len(self.parents) < 1:
            self.depth = 0
            self.parent = None
        elif not hasattr(self, "depth") or self.depth > 0:
            parent_depths = [(parent, self.az.topics[parent].depth) for parent in self.parents]
            if len(parent_depths) < 1:
                self.depth = 0
                self.parent = None
            else:
                self.parent, min_depth = min(parent_depths, key=lambda x: x[1])
                self.depth = min_depth + 1
        return self.depth, self.parent

    def get_content(self):
        if self.html is None: return
        anchor_html = None if self.anchor is None else self.html.find("div", {"id": self.anchor})
        if anchor_html is None: content = self.html.find("div", {"class": "content"})
        else: content = anchor_html
        #else: self.html, content = anchor_html, anchor_html
        if content is None: content = self.html.find("main", {"class": "content"})
        if content is None: content = self.html.find("div", {"id": "main"})
        if content is None: content = self.html.find("div", {"class": "main"})
        # if it still can't find anything... not sure if this is a good idea
        if content is None: content = self.html.find(attrs={"class": "main"})
        if content is None: content = self.html.find(attrs={"class": "content"})
        if content is None: content = self.html.find(attrs={"id": "main"})
        if content is None: content = self.html.find(attrs={"id": "content"})
        if content is None: return
        self.content = content
        self.title = self.html.find("meta", {"name": "title"}).get("content") if self.html.find("meta", {"name": "title"}) else None
        for elem in content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'th', 'td', 'li']):
            if elem.name == 'h1':
                if self.title is None: self.title = elem.text.strip()
                self.additional_topic(elem.text.strip())
            elif elem.name in ['h2', 'h3', 'h4', 'h5', 'h6'] and self.title is None:
                self.title = elem.text.strip()
                self.additional_topic(elem.text.strip())
        if self.title is None or len(self.title) < 1: self.title = self.topic

    def get_subpages(self):
        if self.html is None or (self.sub_pages is not None and len(self.sub_pages) > 0): return
        if self.content is None: self.content = self.html
        self.sub_pages = []
        self.extract_sub_links_from_header()
        logging.info(f"Extracting Page Subpages from {self.link}")
        for sub_page_type in [
            {"elem_type": "nav", "div_class": "anchors"},
            {"elem_type": "article", "div_class": "teaser"},
            {"div_class": "frame--type-menu_teaser_sitemap", "button_class": "solr__listItemLink"},
            {"div_class": "news_list--newsticker"},
            {"div_class": "keywords"},
            {"div_class": "studypaths__list", "button_class": "studypaths__listItem"},
            {"elem_type": "ul", "div_class": "keywords"},
            {"elem_type": "nav", "div_class": "solr-pagination"}
        ]:
            elem_class = sub_page_type["div_class"]
            elem_type = "div" if "elem_type" not in sub_page_type else sub_page_type["elem_type"]
            button_class = None if "button_class" not in sub_page_type else sub_page_type["button_class"]
            for div in self.content.find_all(elem_type, {"class": elem_class}):
                if button_class is not None:
                    buttons = div.find_all("a", class_=button_class)
                    if len(buttons) < 1: buttons = div.find_all("div", class_=button_class)
                else:
                    buttons = div.find_all("a")
                for button in buttons:
                    if button.name != "a": button = button.find("a")
                    if button is None or not button.has_attr('href') or len(button["href"]) < 1: continue
                    if button.find_parent(lambda t: any("contact__" in c for c in t.get("class", []))): continue
                    if button.find_parent(lambda t: any("mail" in c for c in t.get("class", []))): continue
                    self.add_subpage_from_a(button)

    def add_subpage_from_a(self, button):
        sub_page_link = get_link(button["href"])
        if sub_page_link == self.link: return
        if sub_page_link in self.sub_pages or "javascript" in sub_page_link: return
        if button.has_attr('title') and len(button["title"].strip()) > 0:
            sub_page_topic = button["title"].strip()
        elif len(button.text) > 0:
            sub_page_topic = re.sub(r'\s+', ' ', button.text).strip()
        else:
            sub_page_topic = None
        if sum(map(str.isalpha, sub_page_link)) < 2: return
        logging.info(f"Adding Subpage {sub_page_link} from {self.link}")
        self.sub_pages.append(self.az.add_topic(sub_page_link, self.link, sub_page_topic))

    def extract_sub_links_from_header(self):
        if self.anchor is not None: return
        # first check if our page is the title / main page
        nav = self.html.find("div", class_="header__secondRow")
        if nav is not None: nav = nav.find("nav", class_="header__navigationWrapper--sub")
        if nav is None: return
        title = nav.find("div", class_="navigation__title")
        if title is not None: title = title.find("a", href=True)
        logging.info(f"Extracting Header Subpages from {self.link}")
        match_parent = None
        if title is not None and self.link.endswith(title["href"]):
            match_parent = nav
        else:
            nav_ul = nav.find("ul", class_="navigation")
            if not nav_ul: return

            for a in nav_ul.find_all("a", href=True):
                if self.link.endswith(a['href']):
                    match_parent = a.find_parent("li")
                    break

            if match_parent is not None: match_parent = match_parent.find("div", class_="navigation__subWrapper")
        if match_parent is None: return
        sub = match_parent.find("ul", class_="navigation")
        if not sub: return
        for li in sub.find_all("li", recursive=False):
            a = li.find("a", href=True)
            if a is None: continue
            self.add_subpage_from_a(a)
