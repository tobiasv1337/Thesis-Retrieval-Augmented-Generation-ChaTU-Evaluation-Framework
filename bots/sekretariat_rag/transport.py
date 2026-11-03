"""Fixed Rasa dispatch: every free-text message goes to the API action."""

from rasa.engine.graph import GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.shared.nlu.training_data.message import Message


@DefaultV1Recipe.register(DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=False)
class ApiTransport(GraphComponent):
    @classmethod
    def create(cls, config, model_storage, resource, execution_context):
        return cls()

    def process(self, messages: list[Message]) -> list[Message]:
        for message in messages:
            intent = {"name": "rag_query", "confidence": 1.0}
            message.set("intent", intent, add_to_output=True)
            message.set("intent_ranking", [intent], add_to_output=True)
        return messages
