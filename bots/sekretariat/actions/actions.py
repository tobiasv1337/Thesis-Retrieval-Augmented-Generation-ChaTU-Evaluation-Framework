from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import pandas as pd
from modules.sekretariat import Arguments

confidence_range = 0.25


class ActionExtractAnswer(Action):

    def name(self) -> Text:
        return "action_extract_answer"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        #logging.info(tracker.latest_message)
        # create arguments tracker for this turn
        args = Arguments(tracker, dispatcher, domain)

        # print intent and confidence for debugging purposes
        args.print_intent_confidence_entities()
        ## get or set original message
        #args.original_message()

        # check and execute default intents
        if args.default_intents(): return args.end()

        # check and update options stack
        if args.check_options_stack(): return args.end()

        # give intent ranking as options, if there is no clear intent
        if args.confidence < 1.0 or args.intent == "further_results":#args.fallback_nlu:
            if args.present_intent_ranking(): return args.end()

        # remember clicked intent
        args.data["clicked"].append(args.intent)

        # do not offer to ask the fallback nlu again if it has already been asked
        args.fallback_nlu = args.data["fallback_nlu"]

        # if original intent slot is set, don't give further options
        if not isinstance(tracker.get_slot('original_intent'), str):
        #if type(tracker.get_slot('original_intent')) != str:
            if args.present_subintents(): return args.end()
        else:
            # reset original intent
            args.set_slot("original_intent", None)

        # get relevant paragraphs and answer
        args.answer()
        # ask if the correct answer has been found and offer alternative
        args.ask_found()
        return args.end()

# logging.info(tracker.latest_message)
#{'intent': {'id': -5572820311992329892, 'name': 'fristen-termine', 'confidence': 0.47838532924652105}, 'entities': [], 'text': 'hey', 'message_id': 'b5fe2c2f05144fce9588d5e998841de0', 'metadata': {}, 'intent_ranking': [{'id': -5572820311992329892, 'name': 'fristen-termine', 'confidence': 0.47838532924652105}, {'id': 8466605885694108155, 'name': 'adressaenderung', 'confidence': 0.224065497517585}, {'id': -355667099339670631, 'name': 'rueckmeldung', 'confidence': 0.08590395748615201}, {'id': 6416445307659210220, 'name': 'semesterbeitraege', 'confidence': 0.070866629481315}, {'id': -3228067169228737596, 'name': 'kontakt-sprechzeiten', 'confidence': 0.039458509534597}, {'id': 7628972378773939091, 'name': 'beurlaubung', 'confidence': 0.025374986231327}, {'id': 4428793644128879965, 'name': 'beitragserstattung', 'confidence': 0.023450575768947}, {'id': -2409599748082491573, 'name': 'krankenversicherungsnachweis', 'confidence': 0.020579855889081}, {'id': -3381193187502367443, 'name': 'mutterschutz-im-studium', 'confidence': 0.017145212739706}, {'id': 1867240017756196262, 'name': 'diplom-studiengang-auslaufend', 'confidence': 0.014769411645829001}], 'response_selector': {'all_retrieval_intents': [], 'default': {'response': {'id': None, 'responses': None, 'response_templates': None, 'confidence': 0.0, 'intent_response_key': None, 'utter_action': 'utter_None', 'template_name': 'utter_None'}, 'ranking': []}}}
