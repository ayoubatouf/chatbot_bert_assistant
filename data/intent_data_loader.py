from data.data_loader_interface import DataLoaderInterface
import pandas as pd


class IntentDataLoader(DataLoaderInterface):
    def __init__(self, intents_json):
        self.intents_json = intents_json
        self.df = pd.DataFrame({"Pattern": [], "Tag": []})
        self.label_to_id = {}
        self.id_to_label = {}

    def load_data(self):
        self.df = self._extract_patterns_and_tags(self.intents_json)
        self.unique_labels = self.df["Tag"].unique().tolist()
        self.unique_labels = [label.strip() for label in self.unique_labels]
        self.id_to_label = {id: label for id, label in enumerate(self.unique_labels)}
        self.label_to_id = {label: id for id, label in enumerate(self.unique_labels)}
        self.df["label_ids"] = self.df["Tag"].map(
            lambda tag: self.label_to_id[tag.strip()]
        )

    def _extract_patterns_and_tags(self, intents_json):
        for intent in intents_json["intents"]:
            for pattern in intent["patterns"]:
                sentence_tag = [pattern, intent["tag"]]
                self.df.loc[len(self.df.index)] = sentence_tag
        return self.df
