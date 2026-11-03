from rasa_sdk.events import SlotSet
from typing import Optional
import pandas as pd
import requests
import re
import os
import codecs
import markdownify
from bs4 import BeautifulSoup
from markdown import markdown
import logging
import json
import hashlib
from .keywords import get_shared_keyword_extractor
from .config import Config
from .logging import Logger
from .rag import ChatQA
from .generate_az import get_canonical_link
from .legacy_routing import (
    build_descendants_cache,
    expand_intents_with_descendants,
    merge_fallback_rankings,
    prepare_intent_presentation,
)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="bs4.builder._lxml")

path = Config().get("az_data_dir")
kw = get_shared_keyword_extractor(load_only=False)
legacy_rag_index_path = os.getenv("LEGACY_RAG_INDEX_PATH")
qa = {
    "deutsch": ChatQA(language="deutsch", index_path=legacy_rag_index_path, use_tfidf_keywords=True, keyword_extractor=kw),
    "english": ChatQA(language="english", index_path=legacy_rag_index_path, use_tfidf_keywords=True, keyword_extractor=kw),
}


def strike(text):
    result = '\N{rightwards arrow with hook} ' + text
    # result = ''
    # for c in text: result = result + c + '\u0336'

    return result


def number(x):
    try:
        return float(x)
    except Exception as e:
        return False


def ask_llm(context, queries, max_new_tokens=250):
    url = Config().get("qa_llm_url")#'http://0.0.0.0:5000/generate'
    headers = {'Content-Type': 'application/json'}

    data = {
        'max_new_tokens': 250,
        'context': context,  # 'In the world of AI,',
        'queries': queries  # ['what is the latest trend?', 'who are the major players?']
    }

    # self.logging.info(data)
    try:
        response = requests.post(url, headers=headers, json=data)
        return response.json()["generate"]
    except Exception as e:
        logging.warning(e)


class Arguments:

    def __init__(self, tracker, dispatcher, domain, nlu_threshold=0.99, fallback_nlu_threshold=0.1):

        self.tracker = tracker
        self.dispatcher = dispatcher

        self.rag_pipeline = True
        
        self.logging = Logger(self, "sekretariat")
        self.logging.info(f'Message: "{self.tracker.latest_message["text"]}"')

        self.bots = pd.read_csv(Config().get("ports_file"), index_col="bot").to_dict('index')
        
        self.domain = domain
        self.current_mapping = pd.read_csv(path + "current_sekretariat_mapping.csv")
        self.mapping = pd.read_csv(path + "mapping.csv")
        # Enrich current_mapping with additional columns from mapping.csv for matching rows
        # Get additional columns from mapping that don't exist in current_mapping
        additional_columns = [col for col in self.mapping.columns if col not in self.current_mapping.columns]
        if len(additional_columns) > 0:
            # Merge on 'link' to copy over these columns
            self.current_mapping = self.current_mapping.merge(
                self.mapping[['link'] + additional_columns],
                on='link', how='left'
            )
        self.current_mapping = pd.concat([self.current_mapping, self.mapping.copy()]).drop_duplicates(keep="first")
        self.written_responses = pd.read_csv("templates/responses/sekretariat.csv", index_col="name").to_dict('index')
        self._descendants_cache = self._build_descendants_cache()
        self.descendant_depth = self._get_descendant_depth()
        
        if len(self.domain["intents"][0]) > 1:
            self.valid_intents = self.domain["intents"]
        else:
            self.valid_intents = [list(intent)[0] for intent in self.domain["intents"]]
        self.nlu_threshold = nlu_threshold if not number(self.tracker.get_slot('threshold')) else number(
            self.tracker.get_slot('threshold'))
        self.fallback_nlu_threshold = fallback_nlu_threshold
        self.upper_threshold = 1

        self.threshold = self.nlu_threshold

        self.confidence = self.tracker.latest_message["intent"]["confidence"]
        self.entities = pd.DataFrame(tracker.latest_message["entities"])

        self.language = "deutsch" if self.tracker.get_slot('language') != "english" else "english"
        self.debug = False if self.tracker.get_slot('debug') != "true" else True
        self.returns = []
        self.answers = []

        if self.tracker.get_slot('data') is None or self.tracker.get_slot('data') == "null":
            self.data = {
                "options_stack": [],
                "active_options": False,
                "fallback_nlu": False,
                "clicked": [],
                "history": [],
                "intent_history": []
            }
        else:
            self.data = self.tracker.get_slot('data')
        if self.confidence != 1.0:
            self.data["clicked"] = []
            self.data["fallback_nlu"] = False
        self.intent_ranking = pd.DataFrame(self.tracker.latest_message["intent_ranking"]) \
            if self.confidence != 1.0 or "intent_ranking" not in self.data else pd.DataFrame.from_dict(
            self.data["intent_ranking"])
        self.results_table, self.dialog_state = None, None

        self.intent, self.topic, self.link, self.topic_info = None, None, None, None
        self.set_intent(self.tracker.latest_message["intent"]["name"])
        self.fallback_nlu = self.data["fallback_nlu"]
        metadata = self.tracker.latest_message.get("metadata") or {}
        self.legacy_retrieval_capture_enabled = metadata.get("legacy_retrieval_capture") is True
        self.legacy_retrieval_capture_id = str(metadata.get("legacy_retrieval_capture_id") or "")
        self.legacy_retrieval_capture = None
        self.legacy_retrieval_capture_emitted = False
        self.message = self.original_message()

    def _get_descendant_depth(self) -> Optional[int]:
        """Resolve maximum descendant depth from slot/config; None means unlimited."""
        # NOTE: We intentionally do NOT read this from a Rasa slot.
        # Descendant depth is a retrieval tuning knob; persisting it per dialog turn via slots
        # isn't needed for normal bot behavior and also produces warnings when the slot
        # isn't declared in the domain.
        try:
            depth = Config().get("descendant_depth")
        except Exception:
            depth = None
        depth = number(depth) if depth is not None else None
        if isinstance(depth, (int, float)) and depth >= 0:
            return int(depth)
        return 5

    def _build_descendants_cache(self) -> dict[str, dict[str, int]]:
        """Precompute descendant intents with depth for faster lookups per request."""
        return build_descendants_cache(self.current_mapping)

    def _expand_intents_with_descendants(self, intents: list[str], max_depth: Optional[int] = None) -> list[str]:
        """Expand intents with cached descendants derived from current mapping."""
        return expand_intents_with_descendants(intents, self._descendants_cache, max_depth=max_depth)

    def print_intent_confidence_entities(self):
        if "confidence" in self.tracker.latest_message["intent"]:
            self.logging.info(f'Recognized Intent: {self.intent} ({self.confidence})')
            if "entities" in self.tracker.latest_message and len(self.tracker.latest_message["entities"]) > 0:
                self.logging.info(f'Recognized Entities: {self.tracker.latest_message["entities"]}')

    def ask_nlu(self, message, nlu="main"):
        if nlu != "main" and Config().get("ignore_bots_with_error_status") and self.bots["az"]["status"] > 0:
            self.logging.info(f"Fallback nlu status is offline.")
            return None
        if len(message.strip()) < 1:
            self.logging.info(f"Message empty, skipping nlu request. Maybe no keywords found?")
            return None
        port = self.bots["sekretariat"]["rasa"] if nlu == "main" else self.bots["az"]["rasa"]
        url = f'http://localhost:{port}/model/parse'
        data = {"text": message}
        try:
            r = requests.post(url, json=data, headers={'content-type': 'application/json'})
            new_intent_ranking = r.json()["intent_ranking"]
            return pd.DataFrame(new_intent_ranking)
        except Exception as e:
            self.logging.warning(f"The communication with {nlu} bot failed: {e}")

    def ask_fallback_nlu(self):
        self.fallback_nlu, self.data["fallback_nlu"] = True, True
        keyword_message = kw.get_keywords(self.message)
        self.logging.info(f"Message: {self.message}")
        self.logging.info(f"Keywords: {keyword_message}")
        if len(keyword_message.strip()) < 1: self.logging.info(f"No keywords found.")
        responses = [df for df in [
            self.ask_nlu(keyword_message),
            self.ask_nlu(self.message, "fallback"),
            self.ask_nlu(keyword_message, "fallback")
        ] if df is not None]
        if len(responses) < 1: return
        self.intent_ranking = merge_fallback_rankings(self.intent_ranking, responses, threshold=self.threshold)
        self.threshold = self.fallback_nlu_threshold
        self.data["intent_ranking"] = self.intent_ranking.to_dict()
        self.logging.info(self.intent_ranking)
        if "intent_history" not in self.data: self.data["intent_history"] = []
        self.data["intent_history"].append(self.intent_ranking.name.values.tolist())

    def get_intent_ranking(self):
        if not self.fallback_nlu and (self.confidence < self.threshold or self.intent == "further_results"):
            return self.ask_fallback_nlu()
        self.data["intent_ranking"] = self.intent_ranking.to_dict()
        if self.confidence != 1: self.logging.info(self.intent_ranking)
        # else: self.data["fallback_nlu"] = False

    def present_intent_ranking(self):
        if self.get_intent_ranking(): return True
        presentation = prepare_intent_presentation(self.intent_ranking, valid_intents=self.valid_intents, fallback_nlu=self.fallback_nlu, threshold=self.threshold, upper_threshold=self.upper_threshold)
        self.intent_ranking = presentation["filtered_ranking"]
        table_intent_ranking = presentation["table_records"]
        intent_ranking = presentation["display_ranking"]
        if not self.fallback_nlu:
            self.data["intent_ranking"] = self.intent_ranking.to_dict()
        if len(intent_ranking) < 1:
            self.utter(keyword="rephrase", button_keywords=["back"] if len(self.data["options_stack"]) > 0 else [])
            self.returns = []
            self.results_table = None
            return True
        elif self.fallback_nlu:
            self.utter(keyword="fallback_nlu")
        self.set_intent(intent_ranking.name.values[0])
        # self.logging.info(intent_ranking)
        if len(intent_ranking) > 1:
            self.turn_intent_ranking_into_table(table_intent_ranking)
            button_keywords = ["none" if self.fallback_nlu else "different"]
            if len(self.data["options_stack"]) > 0: button_keywords.append("back")
            # eigentlich: f"{written_responses['which topic'][language]} ({written_responses['button'][language]}.)"
            self.utter(
                keyword="which_topic",
                button_intents=list(intent_ranking["name"]),
                button_keywords=button_keywords,
                send_result_table=True
            )
            if not self.confidence == 1.0:
                self.set_slot("original_message", self.tracker.latest_message["text"])
            return True

    def set_intent(self, intent):
        self.intent = intent
        self.topic_info = {}
        self.topic, self.link = "", ""
        if intent in ["no_match", "successful", "change", "further_results", "fallback_nlu", "last_choice"]: return
        if intent not in list(self.current_mapping.intent):
            ranking = self.intent_ranking[self.intent_ranking["name"].isin(list(self.current_mapping.intent))]
            if len(ranking) < 1: return
            self.intent = ranking.name.reset_index(drop=True).iloc[0]
        self.topic_info = self.get_topic_info(intent=self.intent, page_content=True)
        if len(self.topic_info) > 0:
            self.topic, self.link = self.topic_info["topic"], self.topic_info["link"]

    def get_topic_titles_for_intents(self, intents):
        """
        Extract topic/button titles for all intents in intent_ranking.
        Returns a dict mapping intent names to their titles (respecting language).
        """
        # Filter mapping by intents
        filtered = self.current_mapping[self.current_mapping["intent"].isin(intents)].copy()
        # Create title column: prefer manual, fallback to extracted
        filtered["title"] = filtered[f"manual_title_{self.language}"].fillna(filtered[f"extracted_title_{self.language}"])
        #filtered["title"] = filtered[f"manual_title_{self.language}"].fillna(filtered[f"generated_title_{self.language}"]).fillna(filtered[f"extracted_title_{self.language}"])
        filtered = filtered.drop_duplicates(keep="first", subset=["intent"]).set_index('intent').to_dict('index')
        titles = []
        for intent in intents:
            if intent not in filtered: continue
            title = filtered[intent]["title"]
            if not isinstance(title, str) or len(str(title).strip()) < 1: continue
            titles.append(filtered[intent]["title"])
        self.logging.info(f"Button Titles: {titles}")
        return titles

    def get_topic_info(self, intent=None, link=None, simple_request=None, page_content=False):
        if link is not None:
            topic_mapping = self.current_mapping[self.current_mapping.link == link]
            if simple_request == "intent" and len(topic_mapping) > 0: return topic_mapping.intent.reset_index(drop=True).iloc[0]
        elif intent is not None:
            topic_mapping = self.current_mapping[self.current_mapping.intent == intent]
        elif self.intent is not None:
            topic_mapping = self.current_mapping[self.current_mapping.intent == self.intent]
        else:
            topic_mapping = self.current_mapping[
                self.current_mapping.intent == self.tracker.latest_message["intent"]["name"]]

        columns = ["intent", "link", "topic", "english_topic", "title", "english_title", "topics", "english_topics",
                   "sub_pages", "link", "mapping"]
        topic_info = {column: "" for column in columns}

        if topic_mapping is None or len(topic_mapping) < 1:
            if simple_request == "topic": 
                return ""
            if simple_request == "intent":
                return intent if intent is not None else ""
            return topic_info

        topic_mapping_dict = topic_mapping.to_dict(orient="records")[0]
        # get the topic
        for topic_kind in ["manual", "extracted"]:#["manual", "generated", "extracted"]:
            topic_key = f"{topic_kind}_title_{self.language}"
            if not isinstance(topic_mapping_dict[topic_key], str) or len(topic_mapping_dict[topic_key]) < 1: continue
            if simple_request == "topic": return topic_mapping_dict[topic_key]
            topic_info["topic"] = topic_mapping_dict[topic_key]
            break
        recognized_link = topic_mapping_dict["link"]
        intent = topic_mapping_dict["intent"]
        if self.results_table is not None:
            # TODO: get info from results table if it exists
            pass
        try:
            if recognized_link in self.mapping.link:
                folder = self.mapping[self.mapping.link == recognized_link].intent.reset_index(drop=True).iloc[0]
                if intent is None: intent = folder
            elif intent is not None and os.path.exists(f"{path}scraper/topics/{intent}/versions.csv"):
                folder = intent
            else:
                canonical_link = get_canonical_link(recognized_link)
                if isinstance(canonical_link, str) and len(canonical_link) > 2 and canonical_link in self.mapping.link:
                    folder = self.mapping[self.mapping.link == canonical_link].intent.reset_index(drop=True).iloc[0]
                    # recognized link came from current_mapping so we should replace it
                    self.current_mapping.loc[self.current_mapping.link == recognized_link, "link"] = canonical_link
                    current_mapping_file_content = pd.read_csv(path + "current_sekretariat_mapping.csv")
                    current_mapping_file_content.loc[self.current_mapping.link == recognized_link, "link"] = canonical_link
                    current_mapping_file_content.to_csv(path + "current_sekretariat_mapping.csv", index=False)
                else:
                    if simple_request == "topic": return topic_info["topic"]
                    if simple_request == "intent": return intent
                    return topic_info
            version = pd.read_csv(f"{path}scraper/topics/{folder}/versions.csv").sort_values(by=["time"], ascending=False).reset_index(drop=True).iloc[0]
            for column in columns:
                if isinstance(topic_info[column], str) and len(topic_info[column]) > 0: continue
                if isinstance(version[column], str) and version[column].strip().startswith("["):
                    version[column] = version[column].strip()[2:-2]
                topic_info[column] = str(version[column]) if version[column] is not None else None
            prefix = "" if self.language == "deutsch" or (not isinstance(topic_info["english_topic"], str) and not isinstance(topic_info["english_topics"], str)) else "english_"
            topic_info["topic"] = topic_info[prefix+"topic"]
            if not isinstance(topic_info["topic"], str) and isinstance(topic_info[prefix + "topics"], str):
                topic_info["topic"] = min(topic_info[prefix + "topics"].split(", "), key=len)
            if simple_request == "topic": return topic_info["topic"]
            topic_info["intent"] = topic_mapping_dict["intent"]
            if not page_content: return topic_info
            for language in ["", "english"]:
                if len(language) > 0:
                    components = [language]
                else:
                    components = []
                # make sure there is a title
                topic_info["_".join(components + ["title"])] = topic_info["_".join(components + ["title"])] if topic_info["_".join(components + ["title"])] is not None else topic_info["_".join(components + ["topic"])]

                topic_info["_".join(components + ["html"])] = ""
                topic_info["_".join(components + ["md"])] = ""
                topic_info["_".join(components + ["text"])] = ""
                topic_info["_".join(components + ["preview"])] = ""
                try:
                    html = codecs.open(f"{path}scraper/topics/{folder}/{'_'.join(['content'] + components)}.html",
                                       'r').read()
                    md = re.sub(r'\n\n+', '\n', markdownify.markdownify(html, heading_style="ATX")).strip()
                    # html = str(BeautifulSoup(markdown(md), features="lxml"))
                    text = BeautifulSoup(html, features="lxml").get_text('\n').replace('\n','\n\n')
                    topic_info["_".join(components + ["html"])] = html
                    topic_info["_".join(components + ["md"])] = md
                    topic_info["_".join(components + ["text"])] = text
                    # content of sidebar preview is set here
                    topic_info["_".join(components + ["preview"])] = text[:200]
                    if len(text) > 200: topic_info["_".join(components + ["preview"])] += " [...]"
                except Exception as e:
                    if language != "english": self.logging.warning(f"Error while parsing {folder}: {e}")
            return topic_info
        except Exception as e:
            self.logging.warning(f"{intent} {e} {recognized_link}")
            self.logging.info(f"{recognized_link} {link}")
            return topic_info

    def utter(self, text="", keyword=None, buttons=None, button_intents=[], buttons_additional=[], button_keywords=[], send_result_table=False):
        final_buttons = buttons if buttons is not None else self.buttons(
            intents=button_intents,
            keywords=button_keywords,
            additional=buttons_additional
        )
        if keyword is not None: text = self.response(keyword)
        if buttons is None and len(button_intents) > 0:
            self.data["options_stack"].append({"text": text, "options": final_buttons})
            self.data["active_options"] = True
        else:
            self.data["active_options"] = False
        if len(final_buttons) > 0: text += f'\n{self.response("button")}'
        buttons = self.strike_through_used_buttons(final_buttons)
        if send_result_table: self.utter_with_results_table(text, buttons)
        else: self.dispatcher.utter_message(str(text), buttons=buttons)

    def utter_with_results_table(self, text, buttons=None):
        self.dispatcher.utter_message(
            str(text),
            custom={"results_table": self.results_table, "dialog_state": self.dialog_state},
            buttons=buttons if buttons is not None else []
        )

    def payload(self):
        return {
            "success": "/successful",
            "wrong": '/further_results{"confidence":"' + str(self.confidence) + '"}',#"/no_match" if self.fallback_nlu else "/fallback_nlu",
            "none": '/further_results{"confidence":"' + str(
                self.intent_ranking[self.intent_ranking.confidence < self.upper_threshold].head(
                    5).confidence.min()) + '"}',
            "back": "/last_choice",
            # "none": "/no_match",
            "different": '/further_results{"confidence":"' + str(
                self.intent_ranking[(self.intent_ranking.confidence < self.upper_threshold) & (self.intent_ranking.confidence > self.threshold)].head(
                    5).confidence.min()) + '"}',#"/fallback_nlu"
        }

    def button(self, title, payload):
        original_message_entity = '{"original_message":"'+self.message+'"}'
        # json dumps uses the wrong quotation marks for rasa
        #original_message_entity = json.dumps({"original_message": self.message})

        if payload.endswith("}"):
            payload = payload[:-1]+","+original_message_entity[1:]
        else:
            payload = payload + original_message_entity
        return {
            "title": title,
            "payload": payload
        }

    def keyword_button(self, keyword):
        return self.button(self.response(keyword), self.payload()[keyword])

    def topic_button(self, intent, used=[]):
        intent = re.sub(r'\{[^}]*\}', '', intent).strip()
        topic = str(self.get_topic_info(intent=intent, simple_request="topic"))
        # if topic looks like this ["content"] for some reason
        if topic.startswith('[') and topic.endswith(']'): topic = topic[2:-2]
        duplicate_found = lambda x: True if x in used or not isinstance(x, str) else False
        if duplicate_found(topic):
            full_topic = self.get_topic_info(intent=intent)
            prefix = "" if self.language == "deutsch" else "english_"
            duplicate = topic
            if isinstance(full_topic[prefix + "topics"], str):
                topics = full_topic[prefix + "topics"].split(", ")
            else: topics = []
            if duplicate_found(topic) and topic in topics and isinstance(full_topic[prefix + "topics"], str) and len(
                    topics) > 1:
                while topic in topics and len(topics) > 1:
                    topics.remove(topic)
                    topic = min(topics, key=len)
            if duplicate_found(topic) and isinstance(full_topic[prefix + "title"], str) and topic != full_topic[
                prefix + "title"]:
                topic = full_topic[prefix + "title"]
            # elif isinstance(full_topic[prefix+"text"], str):
            #    split_text = full_topic[prefix + "text"].strip().replace("\n",".").replace(",",".").split(".")
            #    topic = split_text.pop(0).strip()
            #    while topic in used or len(topic) < 2:
            #        topic = split_text.pop(0).strip()
            if duplicate_found(topic):
                topic = intent.replace("-", " ").title()
        else:
            duplicate = False
        used.append(topic)
        return self.button(topic, f"/{intent}"), used, duplicate

    def get_english(self, intent, property):
        english_property_value = self.get_topic_info(intent=intent)["english_" + property]
        if english_property_value is None or not isinstance(english_property_value, str) or len(
                english_property_value) < 1:
            return self.get_topic_info(intent=intent)[property]
        else:
            return english_property_value

    def buttons(self, intents=None, keywords=None, additional=None):
        if additional is None: additional = []
        if keywords is None: keywords = []
        if intents is None: intents = []
        buttons = []
        known_intents = self.valid_intents
        used = []
        for intent in intents:
            if intent in known_intents and intent in list(self.current_mapping.intent):
                button, used, duplicate = self.topic_button(intent, used)
                if duplicate is not False:
                    for b, previous_button in enumerate(buttons):
                        if previous_button["title"] == duplicate:
                            buttons[b], used, _ = self.topic_button(previous_button["payload"][1:], used)
                buttons.append(button)
        buttons.extend(additional)
        for keyword in keywords:
            buttons.append(self.keyword_button(keyword))
        return buttons

    def set_slot(self, slot, value):
        if type(value) == str: value = value.replace("'","`").replace('"','`')
        self.returns.append(SlotSet(slot, value))

    def end(self):
        if self.legacy_retrieval_capture_enabled and not self.legacy_retrieval_capture_emitted:
            capture = self.legacy_retrieval_capture or self._empty_legacy_retrieval_capture()
            capture["selected_intent"] = self.intent
            capture["nlu_intent"] = str(self.tracker.latest_message["intent"]["name"])
            capture["nlu_confidence"] = float(self.tracker.latest_message["intent"]["confidence"])
            capture["nlu_intent_ranking"] = [
                {"name": str(item["name"]), "confidence": float(item["confidence"]), "rank": rank}
                for rank, item in enumerate(self.tracker.latest_message.get("intent_ranking", []), start=1)
            ]
            self.dispatcher.utter_message(custom={"legacy_retrieval_capture": capture})
            self.legacy_retrieval_capture_emitted = True
        self.set_slot("data", self.data)
        return self.returns

    def _empty_legacy_retrieval_capture(self):
        raw_query = str(self.tracker.latest_message.get("text") or "")
        return {
            "schema_version": "legacy_retrieval_capture_v3",
            "capture_id": self.legacy_retrieval_capture_id,
            "query_text_hash": "sha256:" + hashlib.sha256(raw_query.encode("utf-8")).hexdigest(),
            "language": "en" if self.language == "english" else "de",
            "retrieval_attempted": False,
            "retrieval_intents": [],
            "items": [],
            "context_items": [],
            "empty_context_retry": False,
            "direct_context_used": False,
            "direct_context_intent": None,
            "direct_context_content_hash": None,
            "routing_titles": [],
        }

    @staticmethod
    def _legacy_capture_document(document):
        source = str(document.get("source") or "")
        content = str(document.get("content") or "").strip()
        digest = hashlib.sha256((source + "\0" + content).encode("utf-8")).hexdigest()[:32]
        return {
            "legacy_chunk_id": "legacy_chunk:" + digest,
            "source_path": source,
            "content_hash": "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "score": float(document.get("score")),
            "query_variant": str(document.get("query_variant") or "current"),
            "intent": str(document.get("intent") or os.path.basename(os.path.dirname(source))),
        }

    def _record_legacy_retrieval_capture(self, rag_results, retrieval_intents, direct_context=None, direct_context_intent=None, routing_titles=None):
        if not self.legacy_retrieval_capture_enabled:
            return
        selected = qa[self.language]._select_context_documents(rag_results or {}, max_docs=30)
        retry_used = "empty_context_retry" in (rag_results or {})
        direct_context = str(direct_context or "")
        capture = self._empty_legacy_retrieval_capture()
        capture.update({
            "retrieval_attempted": True,
            "retrieval_intents": [str(intent) for intent in retrieval_intents],
            "items": [self._legacy_capture_document(item) for item in selected],
            "context_items": [self._legacy_capture_document(item) for item in selected[:30 if retry_used else 15]],
            "empty_context_retry": retry_used,
            "direct_context_used": bool(direct_context),
            "direct_context_intent": str(direct_context_intent) if direct_context else None,
            "direct_context_content_hash": "sha256:" + hashlib.sha256(direct_context.encode("utf-8")).hexdigest() if direct_context else None,
            "routing_titles": [str(title) for title in (routing_titles or [])],
        })
        self.legacy_retrieval_capture = capture

    def response(self, keyword):
        return self.written_responses[keyword][self.language]

    def ask_found(self):
        if len(self.answers) > 0:
            if self.confidence < 1.0:
                # if not self.fallback_nlu: self.set_slot("original_message", self.message)
                self.utter(keyword="found", button_keywords=["wrong", "success"], send_result_table=True)
            elif len(self.data["options_stack"]) > 0:
                self.utter(keyword="found", button_keywords=["back", "success"], send_result_table=True)
            # self.data["clicked"] = []
        else:
            self.utter(keyword="rephrase", button_keywords=["back"] if len(self.data["options_stack"]) > 0 else [])

    def default_intents(self):
        if self.intent == "no_match":
            self.utter(keyword="rephrase", button_keywords=["back"] if len(self.data["options_stack"]) > 0 else [])
            return True
        if self.intent == "successful":
            self.utter(keyword="successful")
            return True
        if self.intent == "change":
            if number(self.get_entity_value("threshold")) is not False:
                self.logging.info(f"nlu threshold changed to: {self.get_entity_value('threshold')}")
            else:
                self.logging.info("nothing changed")
            return True
        if self.intent == "further_results":
            below_confidence = self.tracker.get_slot('below_confidence')
            if below_confidence is not None and not pd.isna(below_confidence) and not below_confidence == "nan":
                self.upper_threshold = float(below_confidence)
            elif self.upper_threshold == 1:
                self.logging.info("no upper confidence set")
                self.upper_threshold = self.nlu_threshold
            # self.fallback_nlu = True
            self.set_slot("below_confidence", None)
        if self.intent == "unknown":
            self.utter(keyword="repeat")
            return True

    def check_options_stack(self):
        if self.intent == "last_choice":
            if len(self.data["options_stack"]) < 1:
                self.intent = "further_results"
                return False
            if self.data["active_options"]: self.data["options_stack"].pop(-1)
            self.utter(text=self.data["options_stack"][-1]["text"])
            self.utter(keyword="previous", buttons=self.data["options_stack"][-1]["options"])
            # self.utter(text=self.data["options_stack"][-1]["text"], buttons=self.data["options_stack"][-1]["options"])
            self.data["active_options"] = True
            return True
        elif self.confidence < 1.0:
            self.data["options_stack"] = []
        self.data["active_options"] = False
        # elif len(self.data["options_stack"]) > 0:
        #    self.data["options_stack"][-1]["options"] = [d for d in self.data["options_stack"][-1]["options"] if d["payload"] != f'/{self.intent}']

    def strike_through_used_buttons(self, buttons):
        options = []
        for d in buttons.copy():
            d = d.copy()
            if d["payload"][1:] in self.data["clicked"]:
                # d["title"] = f'~~{d["title"]}~~'
                d["title"] = strike(d["title"])
                print("STRIKE:", d)
            options.append(d)
        return options

    def present_subintents(self):
        if len(self.topic_info) < 1 or "#" in self.link: return False
        sub_pages = self.topic_info["sub_pages"]
        if type(sub_pages) != str: return False
        sub_pages = [self.get_topic_info(link=link, simple_request="intent") for link in sub_pages.split(", ") if
                     link.strip() in list(self.current_mapping.link) and self.get_topic_info(link=link, simple_request="intent") in self.valid_intents]
        if len(sub_pages) < 1: return False
        # TODO: decide if it should pass the whole page to the llm or rag on the page and subpages?
        #self.turn_intent_ranking_into_table([{"name": self.intent, "confidence": self.confidence}])
        self.turn_intent_ranking_into_table([{"name": page, "confidence": self.confidence} for page in [self.intent]+sub_pages], intent=self.intent)
        text, title, link = self.get_text_title_link(self.topic_info)
        first_sentence = \
            f'{self.response("about")} [{title}]({link}).' \
                if self.tracker.latest_message["text"].startswith("/") else \
                f'{self.response("maybe")} "[{title}]({link})"?'
        additional_button, _, _ = self.topic_button(self.intent)
        additional_button["payload"] = f'/{self.intent}' + '{"original_intent":"' + self.intent + '"}'
        additional_button["title"] = self.response("go").upper() + ": " + additional_button["title"]  # .upper()
        button_keywords = ["none" if self.fallback_nlu else "different"]
        if len(self.data["options_stack"]) > 0: button_keywords.append("back")
        self.utter(
            text=f'{first_sentence} {self.response("subtopics")}',
            button_intents=sub_pages,
            buttons_additional=[additional_button],
            button_keywords=button_keywords,
            send_result_table=True
        )
        self.set_slot("original_message", self.message)
        return True

    def get_text_title_link(self, topic_info, type="text"):
        #self.logging.info(f"topic_info: {topic_info}")
        if self.language == "english" and check_dict_value(topic_info, "english_" + type):
            text = topic_info["english_" + type]
        else:
            if type not in topic_info:
                text = ""
            else:
                text = topic_info[type]
        if self.language == "english" and check_dict_value(topic_info, "mapping"):
            link = topic_info["mapping"]
        else:
            link = topic_info["link"]
        if self.language == "english" and (
                isinstance(topic_info["english_topic"], str) and len(topic_info["english_topic"]) > 0 or isinstance(
            topic_info["english_title"], str) and len(topic_info["english_title"]) > 0):
            title = topic_info["english_title"] if isinstance(topic_info["english_title"], str) else topic_info[
                "english_topic"]
        else:
            title = topic_info["title"] if isinstance(topic_info["title"], str) else topic_info["topic"]
        if topic_info["topic"].lower() != title.lower(): title = " | ".join([t for t in [topic_info["topic"], title] if isinstance(t, str) and t.lower() != "nan"])
        return text.strip(), title, link

    def get_relevant_paragraphs(self):
        if len(self.topic_info) < 1:
            self.logging.info(self.intent)
            self.answers.append("")
            return
        # if english, send english topic name OR
        text, title, link = self.get_text_title_link(self.topic_info, "html")
        # if len(text) > 0:
        #    self.topic, self.link = relevant_paragraphs.page_title.values[0], relevant_paragraphs.link.values[0]
        # if
        if self.confidence == 1.0:
            self.turn_intent_ranking_into_table(intent_ranking=[{"name": self.intent, "confidence": "1.0"}], intent=self.intent)
        else:
            self.turn_intent_ranking_into_table(
                intent_ranking=self.intent_ranking.to_dict(orient="records"),
                intent=self.intent
            )
        # TODO: remove this if sidebar and rag are active?
        if len(text) < 1:
            self.answers.append(self.response("try") + f' "[{title}]({link})".')
            self.answers.append(self.response("preview") + ":\n" + self.response("no preview"))
        else:
            limit = 200
            preview_text = ''.join(BeautifulSoup(text[:limit].strip(), features="lxml").findAll(string=True))
            while len(preview_text) < 500:
                if limit >= len(text.strip()): break
                limit *= 2
                preview_text = ''.join(
                    BeautifulSoup(preview_text[:limit].strip(), features="lxml").findAll(string=True))
            preview_text = text  # .replace("\\n", "\n").replace("\n", "\n\n").replace("*", "\*")
            self.answers.extend([
                # self.response("try") + f' "[{title}]({link})".',
                self.response("try") + f' "[{self.response("preview")}:]({link})".',
                self.response("preview") + f':\n {preview_text[:limit].strip()}'# [...]'
            ])
        # self.call_llm(preview_text)

    def answer(self):
        if len(self.topic_info) > 0:
            text, title, link = self.get_text_title_link(self.topic_info)
            self.utter(f'{self.response("about")} [{title}]({link}).' \
                           if self.tracker.latest_message["text"].startswith("/") else \
                           f'{self.response("maybe")} "[{title}]({link})"?')
            self.get_relevant_paragraphs()
        for answer in self.answers:
            self.utter(text=answer.replace("\n", "  \n"), send_result_table=True)

    def original_message(self):
        if self.confidence == 1.0 and self.tracker.latest_message["text"].startswith("/") and isinstance(
                self.tracker.get_slot('original_message'), str):
            self.message = self.tracker.get_slot('original_message')
        elif self.tracker.latest_message["text"].startswith("/") and '{"original_message":"' in self.tracker.latest_message["text"]:
            start_json = self.tracker.latest_message["text"].find("{")
            try:
                self.message = json.loads(self.tracker.latest_message["text"][start_json:])["original_message"]
                self.set_slot("original_message", self.message)
                self.fallback_nlu, self.data["fallback_nlu"] = False, False
            except Exception as e:
                self.logging.warning(f"original_message: {e}")
            self.intent = self.tracker.latest_message["text"][1:start_json]
            if self.intent not in self.valid_intents:
                self.upper_threshold = 0.9999999
                self.logging.info(self.tracker.get_slot('below_confidence'))
                self.intent = "further_results"
        elif not self.tracker.latest_message["text"].startswith("/"):#self.confidence < 1.0:
            self.set_slot("original_message", self.tracker.latest_message["text"])
            self.message = self.tracker.latest_message["text"]
            self.fallback_nlu, self.data["fallback_nlu"] = False, False
            if "history" not in self.data: self.data["history"] = []
            self.data["history"].append({"role": "user", "content": self.message})
            self._cap_chat_history()
        else:
            self.message = "/unknown"
            self.intent = "unknown"
        return self.message

    def _cap_chat_history(self):
        """Keep chat history bounded to avoid oversized LLM prompts in long conversations."""
        try:
            history = self.data.get("history") or []
            if not isinstance(history, list) or not history:
                return

            # Defaults aim to "accept most" while still preventing pathological growth.
            # Preference order:
            # - config.json/config_local.json (if keys exist)
            # - environment variables (optional)
            # - hardcoded defaults
            cfg = Config()
            max_messages = int(
                cfg.get(
                    "sekretariat_history_max_messages",
                    os.getenv("SEKRETARIAT_HISTORY_MAX_MESSAGES", "80"),
                )
            )  # ~40 turns
            max_total_chars = int(
                cfg.get(
                    "sekretariat_history_max_total_chars",
                    os.getenv("SEKRETARIAT_HISTORY_MAX_TOTAL_CHARS", "30000"),
                )
            )
            max_msg_chars = int(
                cfg.get(
                    "sekretariat_history_max_msg_chars",
                    os.getenv("SEKRETARIAT_HISTORY_MAX_MSG_CHARS", "8000"),
                )
            )

            # Truncate individual entries first.
            for m in history:
                if not isinstance(m, dict):
                    continue
                c = m.get("content", "")
                if isinstance(c, str) and len(c) > max_msg_chars:
                    m["content"] = c[:max_msg_chars] + "\n\n[...truncated...]"

            # Keep only the tail.
            orig_len = len(history)
            if orig_len > max_messages:
                history = history[-max_messages:]

            # Enforce total char budget by dropping oldest until within budget.
            def total_chars(h):
                return sum(len((x or {}).get("content", "") or "") for x in h if isinstance(x, dict))

            orig_chars = total_chars(history)
            while history and total_chars(history) > max_total_chars:
                history = history[1:]

            new_chars = total_chars(history)
            self.data["history"] = history

        except Exception:
            pass

    def get_entity_value(self, type):
        if len(self.entities) < 1 or type not in list(self.entities.entity):
            return None
        return list(self.entities[self.entities.entity == type].value)[0]

    """
    turns intent ranking into table and generates llm response if rag pipeline is active
    """
    def turn_intent_ranking_into_table(self, intent_ranking:list[dict[str, str]], intent:str=None):

        if len(intent_ranking) < 1 and intent is None: return {}

        if self.rag_pipeline:
            direct_context = None
            direct_context_intent = None
            routing_titles = []
            if len(intent_ranking) == 1 or intent is not None:
                intent = intent if intent is not None else intent_ranking[0]["name"]
                context = self.get_topic_info(intent=intent, page_content=True)["text"]
                direct_context = context if len(context) > 0 else None
                direct_context_intent = intent if direct_context else None
                retrieval_intents = self._expand_intents_with_descendants(
                    [intent_obj["name"] for intent_obj in intent_ranking],
                    max_depth=self.descendant_depth
                )
                llm_response, rag_results = qa[self.language].ask_with_direct_context(
                    question=self.message,
                    intent=intent,
                    context=context if len(context) > 0 else None,
                    intent_ranking=retrieval_intents,
                    chat_history=self.data["history"] if "history" in self.data else [],
                    retrieval_only=self.legacy_retrieval_capture_enabled
                )
            else:
                legal_intents = [intent["name"] for intent in intent_ranking]
                topic_titles = self.get_topic_titles_for_intents(legal_intents)
                routing_titles = topic_titles[:5]
                max_history_depth = 3
                if "intent_history" not in self.data: self.data["intent_history"] = []
                for intent_history in self.data["intent_history"][-max_history_depth:]:
                    legal_intents.extend(intent_history)
                legal_intents = list(set(legal_intents))
                retrieval_intents = self._expand_intents_with_descendants(
                    legal_intents,
                    max_depth=self.descendant_depth
                )
                llm_response, rag_results = qa[self.language].ask(
                    question=self.message,
                    intent_ranking=retrieval_intents,
                    chat_history=self.data["history"] if "history" in self.data else [],
                    topic_titles=topic_titles[:5],
                    retrieval_only=self.legacy_retrieval_capture_enabled
                    )

            self._record_legacy_retrieval_capture(
                rag_results,
                retrieval_intents,
                direct_context,
                direct_context_intent,
                routing_titles,
            )

            if llm_response is not None and len(llm_response) > 0:
                if "history" not in self.data: self.data["history"] = []
                self.data["history"].append({"role": "assistant", "content": llm_response})
                self._cap_chat_history()
                self.utter(self.response("generated") + "\n> "+ llm_response.replace("\n\n","\r>\r> ").replace("\n","\n> ") + "\r\r" + self.response("generated_hint"))
            else:
                pass
                #self.utter(self.response("no_rag"))

        else:
            llm_response, rag_results = "", {}

        if len(intent_ranking) < 1: return rag_results

        # Convert multi-variant results to per-intent format (if needed)
        if any(k in rag_results for k in ["current", "last_1", "last_2", "last_3"]):
            aggregated = {}
            for variant_id, variant_data in rag_results.items():
                for doc in variant_data.get("documents", []):
                    intent_name = doc.get("intent")
                    score = doc.get("score")
                    if intent_name and score is not None:
                        if intent_name not in aggregated or score < aggregated[intent_name]["score"]:
                            aggregated[intent_name] = {"content": doc.get("content", ""), "score": score, "rank": 0, "variant": variant_id}
            # Set ranks
            for rank, (intent_name, _) in enumerate(sorted(aggregated.items(), key=lambda x: x[1]["score"]), 1):
                aggregated[intent_name]["rank"] = rank
            aggregated_results = aggregated
        else:
            aggregated_results = rag_results

        self.results_table = []
        for intent in intent_ranking:
            full_result = self.get_topic_info(intent=intent["name"], page_content=True)
            result = {"link": full_result["link"]}
            for column in ["title", "preview"]:
                if column not in full_result: result[column] = ""
                else: result[column] = full_result[column] if (self.language == "deutsch" or "english_"+column not in full_result or len(full_result["english_"+column]) < 1) else full_result["english_"+column]
            if full_result["topic"].lower() != result["title"].lower(): result["title"] = " | ".join([t for t in [full_result["topic"], result["title"]] if isinstance(t, str) and t.lower() != "nan"])
            #result["name"] = intent["name"]
            if intent["confidence"] is not None: result["confidence"] = str(intent["confidence"])
            if intent["name"] in aggregated_results:
                result["preview"] = aggregated_results[intent["name"]]["content"]
                result["score"] = str(aggregated_results[intent["name"]]["score"])
                result["rank"] = str(aggregated_results[intent["name"]]["rank"])
                # Add variant column if available
                if "variant" in aggregated_results[intent["name"]]:
                    result["variant"] = aggregated_results[intent["name"]]["variant"]
            self.results_table.append(result)
        self.dialog_state = {
            "message": self.message,
            "result_count": len(self.results_table),
            "llm_response": llm_response.replace("\n","\n> ")
        }
        #print("RAG Results:\n", rag_results)
        return rag_results

def check_dict_value(info, value):
    if value in info and isinstance(info[value], str) and len(info[value]) > 0: return True
    return False
