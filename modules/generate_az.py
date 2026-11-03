import pandas as pd
import markdownify
from markdown import markdown
import re
from bs4 import BeautifulSoup
import logging
import os

from .translator import translate_api


def get_examples(main=True, backup_args=None):

    from .config import Config
    path = Config().get("az_data_dir")
    df, intents = generate_main_nlu_data(path) if main else generate_backup_nlu_data(path, args=backup_args)
    utterances = df.groupby('intent')['utterance'].apply(list).to_dict()
    if intents is not None:
        for intent in intents:
            if intent in utterances: continue
            utterances[intent] = []
    return utterances


def get_examples_main():
    return get_examples()


def get_examples_backup(
        threshold_keywords=None,
        threshold_kw_training=None,
        depth_keywords=None,
        depth_intents=None,
        depth_training=None,
        toggle_keywords=None,
        toggle_sentences=None,
        toggle_titles=None,
        toggle_original=None,
        toggle_stripped=None,
        toggle_extracted=None,
        toggle_generated=None,
        ignore_anchors=None
):
    '''

    :param threshold_keywords: tf_idf score threshold
    :param threshold_kw_training: tf_idf score threshold
    :param depth_keywords: max-depth used for keyword generation
    :param depth_intents: max-depth used for training
    :param depth_training: max-depth difference that children training data is added to the parent training data (after the depth-intent cut)
    :param toggle_keywords: bool -> use the extracted keywords in the training data
    :param toggle_sentences: bool -> use the page sentences in the training data
    :param toggle_titles: bool -> use the page titles, topics and intent names in the training data
    :param toggle_original: bool -> use original version of sentences and titles in the training data
    :param toggle_stripped: bool -> use keyword version of sentences and titles in the training data
    :param toggle_extracted: bool -> use extracted (non-generated) version in the training data
    :param toggle_generated: bool -> use llm generated versions in the training data
    :return:
    '''
    args = {}
    if threshold_keywords is not None: args["threshold_keywords"] = threshold_keywords
    if threshold_kw_training is not None: args["threshold_kw_training"] = threshold_kw_training
    if depth_keywords is not None: args["depth_keywords"] = depth_keywords
    if depth_intents is not None: args["depth_intents"] = depth_intents
    if depth_training is not None: args["depth_training"] = depth_training
    if toggle_original is not None: args["toggle_original"] = toggle_original
    if toggle_stripped is not None: args["toggle_stripped"] = toggle_stripped
    if toggle_extracted is not None: args["toggle_extracted"] = toggle_extracted
    if toggle_generated is not None: args["toggle_generated"] = toggle_generated
    if toggle_keywords is not None: args["toggle_keywords"] = toggle_keywords
    if toggle_sentences is not None: args["toggle_sentences"] = toggle_sentences
    if toggle_titles is not None: args["toggle_titles"] = toggle_titles
    if ignore_anchors is not None: args["ignore_anchors"] = ignore_anchors
    return get_examples(main=False, backup_args=args)


def generate_main_nlu_data(path, file="manual_utterances"):

    manual_utterances = pd.read_csv(os.path.join(path, file + ".csv"))
    mapping = pd.read_csv(path + "mapping.csv")
    mapping = mapping.drop_duplicates(subset=['link'], keep="first")
    missing_links = manual_utterances.loc[~manual_utterances["link"].isin(mapping["link"].unique()), "link"].unique()
    new_links = find_new_links(missing_links)
    for link in new_links: manual_utterances.loc[manual_utterances["link"] == link, "link"] = new_links[link]
    link_to_intent = mapping.set_index('link')['intent'].to_dict()
    manual_utterances['intent'] = manual_utterances['link'].map(link_to_intent)
    manual_utterances.to_csv(os.path.join(path, file+".csv"), index=False)
    logging.info(f"Start Translation Of Utterances")
    translated_utterances = manual_utterances.copy()
    translated_utterances["utterance"] = translate_api(
        translated_utterances["utterance"].astype(str).tolist(),
        cache_path=path,
    )
    manual_utterances = pd.concat([manual_utterances, translated_utterances]).drop_duplicates(keep="first")
    logging.info(f"Translation Of Utterances Done!")
    # Write mapping to a staged file. Promotion to `current_sekretariat_mapping.csv`
    # should happen only after a successful `rasa train` run.
    mapping[["etag", "intent", "link", "mapping"]].to_csv(path + "current_sekretariat_mapping.staged.csv", index=False)
    return manual_utterances, list(mapping["intent"])

def find_new_links(links):
    '''
    :param links: list of links that are outdated
    :return: a dictionary that maps old links to new links
    '''
    canonical_links_mapping = {}
    for link in links:
        canonical_link = get_canonical_link(link)
        if canonical_link is None:
            logging.info(f"No canonical link found: {link}")
        else:
            canonical_links_mapping[link] = canonical_link
    return canonical_links_mapping

def get_canonical_link(link):
    import requests
    try:
        r = requests.get(link)
    except Exception as e:
        logging.warning(f"Couldn't access link {link}: {e}")
        return None
    html = BeautifulSoup(r.text.strip(), 'html.parser')
    split = link.split("#")
    anchor = split[-1] if len(split) > 1 else None
    canonical_link = html.find("link", {"rel": "canonical"})
    if canonical_link is None: return None
    canonical_link = canonical_link["href"]
    if anchor is not None: canonical_link = f"{canonical_link}#{anchor}"
    if link != canonical_link and len(canonical_link.strip()) > 0:
        logging.info(f"Canonical link retrieved. Changing {link} to {canonical_link}")
        return canonical_link
    elif link == canonical_link:
        logging.info(f"Link is canonical, no change made: {link}")
        return link
    else:
        return None


def generate_backup_nlu_data(path, args=None):

    logging.info(f"Start Load Keyword Extractor")
    from .keywords import AZKeywordExtractor

    if args is None: args = {}
    threshold_keywords = args["threshold_keywords"] if "threshold_keywords" in args else 0.1
    threshold_kw_training = args["threshold_kw_training"] if "threshold_kw_training" in args else 0.05
    depth_keywords = args["depth_keywords"] if "depth_keywords" in args else 999999999
    depth_intents = args["depth_intents"] if "depth_intents" in args else 999999999
    depth_training = args["depth_training"] if "depth_training" in args else 0

    toggle_keywords = args["toggle_keywords"] if "toggle_keywords" in args else True
    toggle_sentences = args["toggle_sentences"] if "toggle_sentences" in args else True
    toggle_titles = args["toggle_titles"] if "toggle_titles" in args else True

    toggle_original = args["toggle_original"] if "toggle_original" in args else True
    toggle_stripped = args["toggle_stripped"] if "toggle_stripped" in args else True

    toggle_extracted = args["toggle_extracted"] if "toggle_extracted" in args else True
    toggle_generated = args["toggle_generated"] if "toggle_generated" in args else False

    ignore_anchors = args["ignore_anchors"] if "ignore_anchors" in args else True

    if (
            not (toggle_original or toggle_stripped) or
            not (toggle_keywords or toggle_sentences or toggle_titles) or
            not (toggle_extracted or toggle_generated)
    ):
        logging.error("When everything is turned off, there is no training data.")

    kw = AZKeywordExtractor(
        load=False, threshold=threshold_keywords, depth=depth_keywords
    )

    logging.info(f"Load Keyword Extractor Done! AZKeywordExtractor was loaded successfully!")

    logging.info(f"Start Extract Sentences and Keywords")

    # mapping link -> intent (input parent link to receive parent intent)
    # we will change the intent it is mapped to to the parent's parent, etc. if the parent is also outside of range
    parent_link_to_intent = kw.mapping[["link", "intent"]].set_index(["link"]).to_dict(orient="index")

    sentences_dfs = []

    print(args)
    for language in kw.languages:
        data = kw.data[kw.data["language"] == language].copy().reset_index()
        values, names = kw.get_transform(data["text"], language)

        # start with smallest depth so when the depth too deep, the depth before it will still be valid
        for d, da in data.sort_values(by=["depth"]).iterrows():
            if ignore_anchors and isinstance(da.get("link"), str) and "#" in da["link"]:
                continue
            intent = da["intent"]
            if da["depth"] > depth_intents:
                if da["depth"] - depth_training > depth_intents: continue
                # if the depth is too high BUT it's still in training range we need to find the closest legal parent
                intent = parent_link_to_intent[da["parent"]]
                # if a future child is mapped to this intent, assign it to the parent
                parent_link_to_intent[da["link"]] = intent
            if da["text"] is None or da["text"] == "None": continue
            curr_values = values[d]
            values_clean = curr_values[curr_values > threshold_kw_training]
            names_clean = names[curr_values > threshold_kw_training]
            sentences = []
            if toggle_extracted:
                # add sentences
                if toggle_sentences: sentences += list(into_sentences(da["text"], language))
                # add keywords
                if toggle_keywords: sentences += list(names[curr_values > threshold_keywords])
                # add topics
                if toggle_titles: sentences += da.topics + [da.intent.replace("-", " ")]
            if toggle_generated:
                postfix = "" if language == "deutsch" else "_english"
                # add sentences
                if toggle_sentences:
                    summary_path = os.path.join(kw.path, "scraper/topics", intent, f"summary{postfix}.txt")
                    if os.path.exists(summary_path):
                        with open(summary_path, "r", encoding="utf-8") as f:
                            summary = f.read().strip()
                            sentences += list(into_sentences(summary, language))
                # add keywords
                if toggle_keywords:
                    keywords_path = os.path.join(kw.path, "scraper/topics", intent, f"keywords{postfix}.csv")
                    if os.path.exists(keywords_path):
                        keywords = pd.read_csv(keywords_path)
                        sentences += list(keywords["keyword"].dropna().str.strip())

                # add topics -> generated topics are our questions right now
                if toggle_titles:
                    questions_path = os.path.join(kw.path, "scraper/topics", intent, f"questions{postfix}.csv")
                    if os.path.exists(questions_path):
                        questions = pd.read_csv(questions_path)
                        sentences += list(questions["question"].dropna().str.strip())
            df = pd.DataFrame({
                "intent": [intent] * len(sentences),
                "language": [da["language"]] * len(sentences),
                "utterance": sentences,
                "type": "original"
            })
            df = df[~df.utterance.isna()]
            df = df.loc[df.utterance.astype(str).str.count('[a-zA-Z]') > 2]
            if len(df) < 1: continue
            extracted_df = df.copy()
            extracted_df["type"] = "extracted"
            # add version of sentences that only contains the keywords (removing all stopwords)
            extracted_df[["score", "utterance"]] = extracted_df.apply(
                lambda x: get_stripped_sentence(x.utterance, names_clean, values_clean), axis=1, result_type="expand")
            # score every sentence by the sum of the values of each word (if the word's value is above the threshold at all)
            df["score"] = extracted_df["score"]

            if toggle_stripped and toggle_original: df = pd.concat([df, extracted_df])
            elif toggle_original: df = df
            elif toggle_stripped: df = extracted_df
            df = df[df.score > 0]
            df = df.loc[df.utterance.str.count('[a-zA-Z]') > 2]
            # weight: every keyword contained in the sentences is summed and divided by the length of the sentence
            df["weight"] = df.utterance.apply(lambda x: get_score(x, names_clean, values_clean))
            # remove sentences without any value
            df = df[df.weight > 0]
            df["intent"] = intent
            sentences_dfs.append(df.drop_duplicates(subset=["utterance"], keep='first'))

    logging.info(f"Extract Sentences and Keywords Done!")

    all_sentences = pd.concat(sentences_dfs).reset_index(drop=True)
    all_sentences.sort_values(by=["score"], ascending=False)

    all_sentences.to_csv(path+"generated_utterances.csv", index=False)
    return all_sentences, None


def replace(text, replace_this, replace_with=""):
    for remove in replace_this:
        while remove in text:
            text = text.replace(remove, replace_with)
    return text


def decode_training_data(text):
    text = text.replace("\n", ".\n")
    text = replace(text, replace_this=["..\n",". \r.",".\n\n","\n \n","\n\n",'\\n',":", "... ",".. ", ". .", ". . "], replace_with=".\n")
    text = replace(text, replace_this=["     ","    ","   ","  ","//","/","|","(",")"], replace_with=" ")
    text = replace(text, replace_this=[",,","*","•","'",'"',"\\r","\r"], replace_with="")
    text = re.sub(r'  +', ' ', text)
    return text


def into_sentences(text, language):

    from .tokenizer import tokenizer

    sentences = []
    for sentence in tokenizer[language].tokenize(decode_training_data(text)):
        sentences.extend([s[:512] for s in sentence.split("\n")])
    df = pd.DataFrame(sentences, columns=["sentences"])
    df.sentences = df.loc[df.sentences.str.count('[a-zA-Z]') > 16].sentences
    df.sentences = df.sentences.str.replace('\.+$', '', regex=True)
    df.sentences = df.sentences.str.strip()
    return list(df.sentences)


def get_score(text, names, values):
    score = 0
    text_clean = re.sub('[^a-zA-ZäöüÄÖÜß\s]+', '', text)
    for value, name in zip(values, names):
        if name in text_clean.lower():
            score += value
    return score/len(text)


def get_stripped_sentence(text, names, values):
    score = 0
    new_text = ""
    names = list(names)
    text_clean = re.sub('[^a-zA-ZäöüÄÖÜß\s]+', '', text)
    for word in text_clean.split(" "):
        if word.lower() in names:
            score += values[names.index(word.lower())]
            new_text += " " + word
    return score, new_text.strip()
