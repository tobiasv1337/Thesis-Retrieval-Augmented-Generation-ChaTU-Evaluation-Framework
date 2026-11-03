from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from modules.master import Arguments
#from rasa.core.events import Restarted


class ActionReact(Action):

    def name(self) -> Text:
        return "action_react"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        args = Arguments(tracker, dispatcher, domain)
        if args.bot != "chatu":
            # Je nachdem was der Intent war kann dann entsprechend auf den Bot reagiert werden.
            # Der Intent sollte immer so heißen wie der Bot und der Ordner, dann können wir den Port leicht aus ports.csv auslesen.
            if args.confidence == 1 or args.directly_addressed or not args.multiple_possible_bots(): args.let_module_answer()
            return args.end()
        # get highest of non-bot intents
        intent = args.ranking[args.ranking.index.isin(list(args.master_functions))].index[0]
        args.master_functions[intent]()
        return args.end()


class ActionRestart(Action):

    def name(self) -> Text:
      return "action_restart"

    async def run(self, dispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # send restart event to all other bots
        args = Arguments(tracker, dispatcher, domain)
        args.reset()

        #return [Restarted()]
        return []
