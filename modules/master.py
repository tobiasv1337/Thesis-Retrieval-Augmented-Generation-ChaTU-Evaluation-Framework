from rasa_sdk.events import SlotSet
import pandas as pd
import requests
import json
import markdown
import difflib
import regex as re
from fuzzywuzzy import process
from .config import Config
from .logging import Logger
import os
import copy

master_names = ["chatu", "master", "all", "alle"]


def get_closest_match(input, possible_values):
    letters = ''.join(filter(str.isalpha, input)).lower()
    close_matches = difflib.get_close_matches(letters, possible_values)
    if len(close_matches) > 0:
        return close_matches[0]
    else:
        return None

def safe_json_value(v):
    return json.dumps(v)[1:-1] if isinstance(v, str) else str(v)


def clean_unreplaced_fields(json_str: str) -> str:
    # Remove lines with values that are still "<something>"
    cleaned = re.sub(r'\s*"[^"]*":\s*"<[^>]+>",?\n?', '', json_str)

    # Remove any trailing commas in the object
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
    return cleaned


class Arguments:
    returns = []

    def __init__(self, tracker, dispatcher, domain):

        self.tracker = tracker
        self.dispatcher = dispatcher
        self.domain = domain

        self.logging = Logger(self, "chatu")
        self.logging.debug(f"Logging directory: {self.logging.log_dir}")
        self.logging.debug(
            f"Debug: {self.logging.debug_mode}, "
            f"experiment: {self.logging.experiment_mode}, "
            f"participant_id: {self.logging.participant_id}")

        self.config = Config()

        self.bots = pd.read_csv(self.config.get("ports_file"), index_col="bot")
        if self.config.get("ignore_bots_with_error_status"):
            self.bots = self.bots[(self.bots.direct == 1) & (self.bots.status < 1)].to_dict('index')
        else:
            self.bots = self.bots[self.bots.direct == 1].to_dict('index')
        self.responses = pd.read_csv(self.config.get("responses_dir") + "master.csv", index_col="name").to_dict('index')

        self.intent = self.tracker.latest_message["intent"]["name"]
        self.message = self.tracker.latest_message["text"]
        self.confidence = self.tracker.latest_message["intent"]["confidence"]
        self.sender_id = self.tracker.sender_id
        self.ranking = pd.DataFrame(tracker.latest_message["intent_ranking"]).set_index('name')  # .to_dict("index")
        self.entities = pd.DataFrame(tracker.latest_message["entities"])
        self.logging.info(f"Message: {self.message}")
        self.logging.info(f"NLU-Result: {self.intent} ({self.confidence})")
        if not self.entities.empty: self.logging.info(self.entities)

        self.language = self.extract_language(tracker.get_slot('language'))
        self.data = self.tracker.get_slot('data') if self.tracker.get_slot('data') is not None else {}

        self.possible_bots = list(self.bots)
        self.min_confidence_threshold = 0.2
        self.similarity_threshold = 0.2
        self.button_similarity_threshold = 0.99
        self.last_bot_boost = 0.5
        self.selected_bot_boost = 0.5
        self.bot = "chatu"
        self.directly_addressed = False
        self.forwarded_intent = ""
        self.forwarded_confidence = ""
        self.sidebar = None

        self.master_functions = {
            "language": self.utter_switch_language,
            "switch": self.utter_switch_bot,
            "start": self.utter_start,
            "blank": self.utter_blank,
            "help": self.utter_help,
            "who": self.utter_who,
            "feedback_good": self.utter_feedback,
            "feedback_bad": self.utter_feedback
        }

        # check if message is a typed button
        if not self.message.startswith("/"): self.get_most_similar_button()

        self.logging.msg_log(self.tracker.latest_message["text"], "user", ts=self.tracker.events[-1]["timestamp"])
        self.get_bot()
        if self.bot != "chatu": self.bot_reranking()
        self.logging.info(f"# Result: {self.bot} {self.intent} {self.confidence}")
        self.logging.info(f"# Ranking:\n{self.ranking}")

    def utter_start(self):
        self.utter_template("start")

    def utter_blank(self):
        return

    def utter_greeting(self):
        self.utter_template("start")
        buttons = [{'title': "🇩🇪", 'payload': f'/language' + '{"language":"deutsch"}'},
                   {'title': "🇺🇸/🇬🇧", 'payload': f'/language' + '{"language":"english"}'}]
        self.switch_language()
        self.utter_template("start", buttons=buttons)
        self.switch_language()

    def utter_switch_language(self):
        if len(self.entities) < 1 or "language" not in list(self.entities.entity): self.switch_language()
        language_button = {'title': "🇺🇸/🇬🇧",
                           'payload': f'/language' + '{"language":"english"}'} if self.language == "deutsch" else {
            'title': "🇩🇪", 'payload': f'/language' + '{"language":"deutsch"}'}
        self.utter_template("language", buttons=[
            language_button,
            {'title': "Infos", 'payload': f'/who'}])

    def utter_feedback(self):
        feedback = "bad" if self.intent.endswith("bad") else "good"
        if len(self.entities) < 1 or "message_id" not in list(self.entities.entity):
            message_ids = self.logging.get_latest_message_ids()
            feedback_type = "written"
            self.logging.info(f"{feedback.title()} feedback on latest messages.")
        else:
            message_ids = [self.entities[self.entities.entity == "message_id"].value[0]]
            feedback_type = "thumbs"
            self.logging.info(f"{feedback.title()} feedback on a past message.")
        for message_id in message_ids: self.logging.update_feedback(message_id, feedback, feedback_type)
        if self.intent.endswith("good"):
            self.utter_template("feedback_good")
        elif self.intent.endswith("bad"):
            self.utter_template("feedback_bad")

    def utter_help(self):
        bots = None
        if len(self.entities) > 0 and "bot" in list(self.entities.entity):
            bots = [self.extract_bot_name(bot) for bot in list(self.entities[self.entities.entity == "bot"].value) if
                    self.extract_bot_name(bot) is not None]
        if bots is None or len(bots) < 1:
            # one button per bot
            self.utter_template("help_chatu", buttons=[
                {'title': bot.capitalize(), 'payload': f'/help' + '{"bot":"' + bot + '"}'} for bot in
                self.possible_bots])
        else:
            for bot in bots:
                if bot is None: continue
                self.bot = bot
                self.dispatcher.utter_message(text=f"{self.bots[self.bot]['emoji']} {self.bot.upper()}:")
                self.utter_template(f"help_{bot}")

    def utter_who(self):
        bots = None
        if len(self.entities) > 0 and "bot" in list(self.entities.entity):
            bots = [self.extract_bot_name(bot) for bot in list(self.entities[self.entities.entity == "bot"].value) if
                    self.extract_bot_name(bot) is not None]
        if bots is None or len(bots) < 1:
            language_button = {'title': "🇺🇸/🇬🇧",
                               'payload': f'/language' + '{"language":"english"}'} if self.language == "deutsch" else {
                'title': "🇩🇪", 'payload': f'/language' + '{"language":"deutsch"}'}
            self.utter_template("who_chatu", buttons=[
                {'title': self.get_template("go"), 'payload': '/start'},
                {'title': self.get_template("more"), 'payload': '/help'},
                language_button
            ])
        else:
            for bot in bots:
                if bot is None: continue
                self.bot = bot
                self.dispatcher.utter_message(text=f"{self.bots[self.bot]['emoji']} {self.bot.upper()}:")
                self.utter_template(f"who_{bot}", buttons=[
                    {'title': self.get_template("more"), 'payload': '/help' + '{"bot":"' + bot + '"}'}
                ])

    def utter_switch_bot(self):
        self.logging.info(self.possible_bots)
        if len(self.entities) > 0 and "bot" in list(self.entities.entity):
            mentioned_bot = list(self.entities[self.entities.entity == "bot"].value)[0]
            # erstmal nur erster erwähnter bot beachten
            mapped_mentioned_bot = self.extract_bot_name(mentioned_bot)
            if mapped_mentioned_bot is None or mentioned_bot in ["chatu", "alle", "all", "jeder"]:
                self.data["bot"] = "alle"
                self.utter_template("all")
            self.data["bot"] = mapped_mentioned_bot
            self.utter_template("only", self.data["bot"].upper())
        else:
            # TODO: english as well
            buttons = [{'title': "alle", 'payload': f'/switch' + '{"bot":"alle"}'}]
            buttons.extend([{'title': bot.capitalize(), 'payload': f'/switch' + '{"bot":"' + bot + '"}'} for bot in
                            self.possible_bots])
            self.utter_template("switch", buttons=buttons)

    def get_bot(self):
        self.filter_possible_bots()
        if self.confidence < 1: self.filter_possible_intents()
        if "@" in self.message: self.get_addressed_bot_and_message()
        if self.intent == "smalltalk" and "smalltalk" in list(self.bots):
            self.bot = self.intent
            self.confidence = 1
        elif self.intent in self.possible_bots:
            self.bot = self.intent
        else:
            self.bot = "chatu"

    def get_most_similar_button(self):
        setting = self.config.get("match_message_to_buttons")  # options "complete", "last", "false"
        if setting == "false" or not setting:
            return None

        # Get all buttons of the conversation into a pandas dataframe (title & payload)
        events = pd.DataFrame(self.tracker.events)
        if "data" not in events.columns: return None
        buttons_including_none = pd.DataFrame(list(events[events["event"] == "bot"]["data"]))["buttons"]

        if setting == "complete":
            buttons = [pd.DataFrame(b) for b in list(buttons_including_none[~buttons_including_none.isna()])]
            if len(buttons) < 1: return None
            buttons = pd.concat(buttons).drop_duplicates(subset=["title"], keep="last")
        else:
            # Alternatively get only the latest button (strategy setting in config)
            buttons = pd.DataFrame(list(buttons_including_none[~buttons_including_none.isna()])[-1])
        if buttons.empty:
            return None

        # Get the latest user message
        user_messages = events[events["event"] == "user"]
        if user_messages.empty:
            return None
        last_message = user_messages.iloc[-1]["text"]

        # Calculate similarity of message to button titles
        button_titles = buttons["title"].tolist()
        best_match, score = process.extractOne(last_message, button_titles)

        self.logging.info(f"Match message to button: {score}, {best_match}")

        # If similarity is high enough, return payload otherwise None
        if score / 100 < self.button_similarity_threshold: return None
        payload = buttons[buttons["title"] == best_match]["payload"].values[0]
        if payload is None: return None

        self.logging.info(f'Found similar button: "{best_match}" ({score})')
        self.logging.info(f'Message changed from "{self.message}" to "{payload}".')

        self.message = payload
        self.confidence = 1
        self.intent = payload[1:].split("{")[0]

    def filter_possible_intents(self):
        possible_intents = self.possible_bots + list(self.master_functions)
        self.ranking = self.ranking[self.ranking.index.isin(possible_intents)]
        self.ranking = self.ranking[
            (~self.ranking.index.isin(list(self.master_functions))) | (self.ranking.confidence >= 0.9)]
        # self.ranking[self.ranking.index.isin(list(self.master_functions))] -= 0.1
        # if self.intent not in possible_intents:
        self.intent = self.ranking.index[0]

    def bot_reranking(self):
        if self.message.startswith("/"):
            self.get_button_message()
        elif self.confidence < 1:
            self.rerank_bots()

    def get_button_message(self):
        if self.tracker.get_slot('button') is not None:
            self.message = self.tracker.get_slot('button')
        elif '"button"' in self.message:
            entities = json.loads(self.message[self.message.index("{"):])
            if "button" in entities:
                self.message = entities["button"]
        else:
            self.message = ""
            self.logging.info("add debugging option here")
        self.logging.info(f"Button message: {self.message}")

    def get_addressed_bot_and_message(self):
        split_message = self.message.split(" ")
        new_message = []
        addressed_bots = []
        chatu = False
        for word in split_message:
            if word.startswith("@"):
                closest_match = self.extract_bot_name(word)
                if closest_match is None:
                    new_message.append(word)
                elif closest_match in master_names:
                    chatu = True
                else:
                    addressed_bots.append(closest_match)
            else:
                new_message.append(word)
        if len(addressed_bots) < 1 and chatu == False: return
        if len(addressed_bots) > 0 and self.intent in ["who", "help"]:
            self.intent = self.intent + "_" + addressed_bots[0]
            return
        # if len(addressed_bots) > 1:
        self.possible_bots = addressed_bots
        self.message = " ".join(new_message)
        self.redo_intent_ranking()
        if chatu:
            self.filter_possible_intents()
        else:
            if len(self.ranking[self.ranking.index.isin(self.possible_bots)]) > 0:
                self.ranking = self.ranking[self.ranking.index.isin(self.possible_bots)]
            self.intent = self.ranking.index[0]

        self.directly_addressed = True
        # else:
        #    self.bot = addressed_bots[0]
        #    self.intent = addressed_bots[0]
        #    self.confidence = 1

        # TODO: catch smalltalk before all this
        # self.bot = closest_match if ranking_dict[closest_match]["confidence"] > ranking_dict["smalltalk"][
        #    "confidence"] else "smalltalk"
        # else:
        #    #TODO: trigger switch intent ? or try answering anyways and say that it wasn't clear which bot was addressed

    def rerank_bots(self):
        # TODO: if something is really high in ranking now it should be saved and offered as an option later
        self.ranking = self.ranking[self.ranking.index.isin(self.possible_bots)]
        self.logging.info(self.ranking.round(6))
        if "last_bot" in self.data:
            # boost confidence of "last_bot"
            self.ranking.loc[self.ranking.index == self.data["last_bot"], "confidence"] += self.last_bot_boost
            self.logging.info(f'Last bot {self.data["last_bot"].title()} boosted with +{self.last_bot_boost}.')
        if "bot" in self.data and self.data["bot"] is not None and self.data["bot"] != "alle":
            # boost confidence of selected bot
            self.ranking.loc[self.ranking.index == self.data["bot"], "confidence"] += self.selected_bot_boost
            self.logging.info(f'Selected bot {self.data["bot"].title()} boosted with +{self.selected_bot_boost}.')
        # boost bots or punish unfinished / unimportant bots
        for bot in self.bots:
            if "boost" not in self.bots[bot]: continue
            if pd.isna(self.bots[bot]['boost']) or self.bots[bot]['boost'] is None: continue
            if type(self.bots[bot]['boost']) != int and type(self.bots[bot]['boost']) != float: continue
            self.ranking.loc[self.ranking.index == bot, "confidence"] += self.bots[bot]['boost']
            # self.ranking.loc[self.ranking.index == "studyprograms", "confidence"] -= 1

        self.ranking = self.ranking.sort_values(by="confidence", ascending=False)
        self.logging.info(f"New ranking:\n{self.ranking.round(6)}")
        self.bot = list(self.ranking.index)[0]
        if self.confidence < 1: self.confidence = self.ranking.confidence.max()

    def filter_possible_bots(self):
        intents_in_domain = [
            list(intent)[0] if isinstance(intent, dict) else intent for intent in self.domain["intents"]]
        for bot in self.possible_bots:
            if bot in intents_in_domain: continue
            self.possible_bots.remove(bot)

    def get_template(self, type, input=""):
        if type not in self.responses:
            self.logging.info(type)
            self.logging.info("response not defined")
            return self.responses["empty"][self.language]
        return self.responses[type][self.language].replace("{}", input).replace("\\n", "\n")

    def utter_template(self, type, input="", buttons=[]):
        template = self.get_template(type, input)
        self.utter(template, buttons)

    def send(self):
        if self.bots[self.bot]["status"] > 0:
            return [{"text": self.get_template("offline", self.bot.title())}]
        url = f'http://localhost:{self.bots[self.bot]["rasa"]}/webhooks/rest/webhook'
        self.logging.info(url)
        message = self.message
        match = re.search(r'({.*})$', message)

        if match:
            json_part = match.group(1)
            try:
                formatted_json = json.dumps(json.loads(json_part.replace("'", '"')))  # Ensure double quotes
                message = message[:match.start()] + formatted_json  # Replace only the JSON part
            except json.JSONDecodeError as e:
                self.logging.warning(f"Trying to format forwarded message json: {e}")
                pass

        data_json = {
            "sender": self.sender_id,
            "message": message
        }
        self.logging.info(f"To {self.bot}: {data_json}")
        try:
            r = requests.post(url, data=json.dumps(data_json), headers={'Content-Type': 'application/json'})
            return r.json()
        except Exception as e:
            self.logging.error(f"Send request failed: {e}")
        # try:
        #    r = requests.post(url, json=data, headers={'content-type': 'application/json'})
        #    return r.json()
        # except Exception as e: self.logging.warning(f"send: {e}")
        return [{"text": self.get_template("failed", self.bot.title())}]

    def get_confidences(self):
        confidences = {}
        for bot in self.possible_bots:
            _, confidence = self.get_intent_and_confidence(bot)
            if confidence is None: continue
            confidences[bot] = confidence
        return confidences

    def get_intent_and_confidence(self, bot=None):
        if bot is None: bot = self.bot
        url = f'http://localhost:{self.bots[bot]["rasa"]}/model/parse'
        data = {"text": self.message}
        try:
            r = requests.post(url, json=data, headers={'content-type': 'application/json'}).json()
            if "intent" not in r:
                self.logging.info(r)
            return r["intent"]["name"], r["intent"]["confidence"]
        except Exception as e:
            self.logging.warning(f"get_intent_and_confidence: {e}")
            return "", None

    def redo_intent_ranking(self):
        url = f'http://localhost:{self.bots["chatu"]["rasa"]}/model/parse'
        data = {"text": self.message}
        try:
            r = requests.post(url, json=data, headers={'content-type': 'application/json'})
            self.ranking = pd.DataFrame(r.json()["intent_ranking"]).set_index('name')
        except Exception as e:
            self.logging.warning(f"redo_intent_ranking: {e}")

    def set_slots(self, slots):
        self.logging.info("set_slots")
        events = list()
        for slot in slots: events.append({"event": "slot", "name": slot, "value": slots[slot]})
        url = f'http://localhost:{self.bots[self.bot]["rasa"]}/conversations/{self.sender_id}/tracker/events'
        try:
            r = requests.post(url, json=events, headers={'content-type': 'application/json'})
        except Exception as e:
            self.logging.warning(f"set_slots: {e}")

    def create_custom(self, text, buttons=[], app=True, link=None):
        self.logging.info("create_custom")
        message_id = self.logging.msg_log(text, buttons=buttons)
        custom = {"bot": self.bot}
        custom["original"] = {
            "text": text,
            "buttons": buttons
        }
        custom["markdown"] = {
            "text": text,
            "buttons":
                [{
                    "title": button['title']
                    if not "\N{rightwards arrow with hook}" in button['title']
                    else button['title'].replace("\N{rightwards arrow with hook} ", "~~") + "~~",
                    "payload": button['payload']
                } for button in buttons if
                    not isinstance(button, float) and isinstance(button['title'], str) and isinstance(button['payload'],
                                                                                                      str)]
        }
        custom["html"] = {
            "text": markdown.markdown(custom["markdown"]["text"]),
            "buttons": [
                f"<button payload='{button['payload']}'>"
                + markdown.markdown(button['title'], extensions=['pymdownx.tilde'])[3:-4]
                + "</button>" for button in custom["markdown"]["buttons"]]
        }
        if link is not None: custom["link"] = link
        custom["app"] = app
        custom["message_id"] = message_id
        if self.sidebar is not None: custom["sidebar"] = self.sidebar
        '''if self.bot == "smalltalk": custom["sidebar"] = {
            "header": "## 🧪 Full Markdown Demo\n\nAn overview of what Markdown can do:\n- Headings, lists, tables, and links\n- Images and formatting\n- Code blocks, math, checkboxes, footnotes",
            "content": [
                {
                    "img_url": "https://www.static.tu.berlin/fileadmin/www/_processed_/1/1/csm_TUBerlin-ImagebilderOktober2019-PhilippArnoldtPhotography-099_68a3aac319.jpg",
                    "url": "https://www.tu.berlin",
                    "card_description": "### 🔗 Links & Images\nYou can embed images and make them clickable:\n[![TU Berlin](https://www.static.tu.berlin/fileadmin/www/_processed_/1/1/csm_TUBerlin-ImagebilderOktober2019-PhilippArnoldtPhotography-099_68a3aac319.jpg)](https://www.tu.berlin)",
                    "table": {
                        "title": "Images & Links",
                        "id": "0",
                        "possibilities": 2,
                        "fun_fact": "You can combine links and images in one line of markdown!"
                    }
                },
                {
                    "img_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Markdown-mark.svg/1920px-Markdown-mark.svg.png",
                    "url": "https://commonmark.org/help/",
                    "card_description": "### 💬 Text & Formatting\nMarkdown supports:\n- **Bold** and _italic_\n- `inline code`\n- ~~strikethrough~~\n\n> This is a blockquote. Useful for highlighting info.\n\n### Lists\n- Bullet\n- Points\n  - Nested ones\n\n1. Numbered\n2. Lists\n3. Are easy",
                    "table": {
                        "title": "Text & Lists",
                        "id": "1",
                        "possibilities": 6,
                        "fun_fact": "You can nest bullets and numbers as deep as you want."
                    }
                },
                {
                    "img_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Check_green_circle.svg/2048px-Check_green_circle.svg.png",
                    "url": "https://www.markdownguide.org/extended-syntax/#tables",
                    "card_description": "### 📊 Tables\n| Syntax      | Description | Test Text     |\n| ----------- | ----------- | ------------- |\n| Header      | Title       | Here's this   |\n| Paragraph   | Text        | And more      |\n\nTables support alignment too:\n| Left | Center | Right |\n| :--- | :----: | ----: |\n| A    |   B    |     C |",
                    "table": {
                        "title": "Tables",
                        "id": "2",
                        "possibilities": 10,
                        "fun_fact": "Tables are not in original Markdown, but most renderers support them!"
                    }
                },
                {
                    "img_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Information_icon.svg/1024px-Information_icon.svg.png",
                    "url": "https://www.markdownguide.org/basic-syntax/#code",
                    "card_description": "### 💻 Code\nYou can show code like this:\n\n```python\nprint(\"Hello, Markdown!\")\ndef add(a, b):\n    return a + b\n```\n\nOr inline like `print(\"hi\")`.",
                    "table": {
                        "title": "Code Blocks",
                        "id": "3",
                        "possibilities": 3,
                        "fun_fact": "Syntax highlighting depends on the Markdown renderer."
                    }
                },
                {
                    "img_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/Math_example.svg/1920px-Math_example.svg.png",
                    "url": "https://katex.org/docs/supported.html",
                    "card_description": "### 📐 Math Syntax (KaTeX/LaTeX)\nDisplay math:\n\n```latex\n\\int_0^1 x^2 \\, dx = \\frac{1}{3}\n```\n\nInline math: `E = mc^2`, rendered as $E = mc^2$ (if supported).",
                    "table": {
                        "title": "Math Syntax",
                        "id": "4",
                        "possibilities": 5,
                        "fun_fact": "Math in Markdown requires a renderer that supports LaTeX (e.g., KaTeX or MathJax)."
                    }
                },
                {
                    "img_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Checklist_icon.svg/1024px-Checklist_icon.svg.png",
                    "url": "https://www.markdownguide.org/extended-syntax/#task-lists",
                    "card_description": "### ✅ Task Lists\nUse `- [ ]` for tasks:\n\n- [x] Learn Markdown\n- [x] Use task lists\n- [ ] Finish the project\n- [ ] Celebrate 🎉",
                    "table": {
                        "title": "Task Lists",
                        "id": "5",
                        "possibilities": 4,
                        "fun_fact": "GitHub-flavored Markdown supports interactive checkboxes in issues and PRs."
                    }
                },
                {
                    "img_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/64/Icon-annotation.svg/2048px-Icon-annotation.svg.png",
                    "url": "https://www.markdownguide.org/extended-syntax/#footnotes",
                    "card_description": "### 🦶 Footnotes\nAdd extra info without clutter:\n\nMarkdown is great[^1].\n\n[^1]: Here's the footnote text.\n\nUse `[^1]` for the reference and add the definition anywhere.",
                    "table": {
                        "title": "Footnotes",
                        "id": "6",
                        "possibilities": 2,
                        "fun_fact": "Footnotes are not part of original Markdown, but widely supported in modern engines."
                    }
                },
                {
                    "img_url": None,
                    "url": None,
                    "card_description": "### No Image or URL",
                    "table": {
                        "title": "No Image or URL",
                        "id": "7",
                        "possibilities": 0,
                        "fun_fact": "This card is small."
                    }
                },
                {
                    "img_url": None,
                    "url": "https://www.tu.berlin",
                    "card_description": "### No Image",
                    "table": {
                        "title": "No Image",
                        "id": "8",
                        "possibilities": 1,
                        "fun_fact": "This card is small."
                    }
                },
                {
                    "img_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Markdown-mark.svg/1920px-Markdown-mark.svg.png",
                    "url": None,
                    "card_description": "### No URL",
                    "table": {
                        "title": "No URL",
                        "id": "9",
                        "possibilities": 2000,
                        "fun_fact": "This card is small."
                    }
                },
                {
                    "img_url": None,
                    "url": None,
                    "card_description": None,
                    "table": {
                        "title": "Nothing",
                        "id": "10",
                        "possibilities": 0,
                        "fun_fact": "This card shouldn't exist."
                    }
                }
            ],
            "column_order": ["title", "id", "possibilities", "fun_fact"]
        }'''

        return custom

    def utter(self, text, buttons=[]):
        self.logging.info(f"utter: {text[:300]} {'...' if len(text) > 300 else ''}")
        self.dispatcher.utter_message(text, buttons=buttons, custom=self.create_custom(text, buttons))

    def forward_response(self):
        response = self.send()
        self.forwarded_intent, self.forwarded_confidence = self.get_intent_and_confidence()
        if len(response) < 1:
            if self.message is None or self.message.startswith("/"): return
            self.utter_template("empty")

        # ensure the last element has a "text" field (so "custom" isn't last)
        if len(response) > 1 and "text" not in response[-1]:
            for i, r in reversed(list(enumerate(response[:-1]))):
                if "text" in r:
                    response.append(response.pop(i))
                    break

        for r in response:
            if "custom" in r and ("results_table" in r["custom"] or "dialog_state" in r["custom"]):
                self.handle_custom(r["custom"])
            if "buttons" not in r:
                if "text" in r: self.utter(r["text"])
                continue
            buttons = []
            for button in r["buttons"]:
                buttons.append(self.create_forwarding_button(button["title"], button["payload"]))
            # if "http" r["text"]:
            if "text" in r: self.utter(r["text"], buttons)

    def handle_custom(self, custom):
        self.logging.info("handle_custom")
        if "results_table" not in custom and "dialog_state" not in custom: return
        try:
            sidebar = {"header": "", "content": [], "column_order": []}
            # apply template
            with open(os.path.join(self.config.get("templates_dir"), "sidebar", self.bot + ".json")) as f:
                template = json.load(f)

            # get header from "dialog_state"
            if "dialog_state" in custom and custom["dialog_state"] is not None:
                sidebar["header"] = template[self.language]["header"]
                for k, v in custom["dialog_state"].items():
                    sidebar["header"] = sidebar["header"].replace(f"<{k}>", str(v))
            else:
                sidebar["header"] = f"## {self.bot.title()}"
            existing_columns = set()
            if "results_table" in custom and custom["results_table"] is not None:
                self.logging.info(f'Results Table received: {len(custom["results_table"])} elements')
                if "column_order" in custom and custom["column_order"] is not None:
                    column_order = custom["column_order"]
                elif "column_order" in template[self.language]:
                    column_order = template[self.language]["column_order"]
                elif "content" in template[self.language] and "table" in template[self.language]["content"]:
                    column_order = template[self.language]["content"]["table"].keys()
                else: column_order = []
                content_json = json.dumps(template[self.language]["content"], indent=2)
                for result in custom["results_table"]:
                    content = content_json
                    for k, v in result.items():
                        content = content.replace(f"<{k}>", safe_json_value(v))
                        existing_columns.add(k)
                    content = clean_unreplaced_fields(content)
                    sidebar["content"].append(json.loads(content))
                #sidebar["column_order"] = [c for c in column_order if c in existing_columns]
                sidebar["column_order"] = list(column_order)

            # print into file for debug
            with open(f'templates/sidebar/debug_{self.bot}.json', 'w') as f:
                json.dump(sidebar, f, indent=4)
            # Add to custom (custom creation reads self.sidebar
            self.sidebar = sidebar
        except Exception as e:
            self.logging.error(f"Error while parsing sidebar template: {e}")

    def create_forwarding_button(self, title, payload, bot=None):
        if bot is None: bot = self.bot
        return {
            'title': title,
            'payload': f'/{bot}' + '{"button":"' + payload.replace('"', "'") + '"}'
        }

    def extract_bot_name(self, mentioned_bot):
        match = get_closest_match(mentioned_bot, master_names + self.possible_bots)
        return match if match not in master_names else "chatu"

    def extract_language(self, mentioned_language):
        match = get_closest_match(mentioned_language, ["deutsch", "english"])
        return "deutsch" if match is None else match

    def set_slot(self, slot, value):
        self.returns.append(SlotSet(slot, value))

    def end(self):
        self.set_slot("data", self.data)
        self.set_slot("language", self.language)
        return self.returns

    def switch_language(self):
        self.language = "english" if self.language == "deutsch" else "deutsch"

    def multiple_possible_bots(self):
        if self.confidence < self.min_confidence_threshold:
            confidences = self.get_confidences()
            self.logging.info(confidences)
            options = sorted(confidences, key=confidences.get, reverse=True)
            # TODO: utter unsure? or direct to switch
        if self.confidence >= self.min_confidence_threshold or len(options) < 1:
            options = list(self.ranking[self.ranking["confidence"] > self.confidence - self.similarity_threshold].index)
        if len(options) > 1:
            self.logging.info(f"Options: {options}")
            buttons = [self.create_forwarding_button(bot.capitalize(), self.message, bot) for bot in options]
            self.utter_template("addressed", self.bot.upper(), buttons)
            return True
        return False

    def let_module_answer(self):
        self.logging.info({"language": self.language, "debug": self.tracker.get_slot('debug')})
        self.set_slots({"language": self.language, "debug": self.tracker.get_slot('debug')})
        self.dispatcher.utter_message(text=f"{self.bots[self.bot]['emoji']} {self.bot.upper()}:")
        self.forward_response()
        if self.bot not in ["smalltalk", "chatu"]: self.data["last_bot"] = self.bot

    def reset_bots(self):
        self.logging.info(f"Restart Session {self.sender_id}.")
        for bot in self.bots:
            url = f'http://localhost:{self.bots[bot]["rasa"]}/webhooks/rest/webhook'
            data_json = {"sender": self.sender_id, "message": "/restart"}
            try:
                r = requests.post(url, json=data_json, headers={'content-type': 'application/json'})
                return r.json()
            except Exception as e:
                self.logging.warning(f"reset_bots: {e}")
            # return [{"text": self.get_template("failed", self.bot)}]
