from chatbot.simple_chatbot import SimpleChatbot
from helper.utils import load_json_data
from transformers import BertForSequenceClassification, BertTokenizerFast
from transformers import pipeline


if __name__ == "__main__":
    filename = "intents.json"
    intents_data = load_json_data(filename)

    model_path = "pretrained_chatbot_model"
    model = BertForSequenceClassification.from_pretrained(
        model_path, from_tf=False, from_flax=False
    )

    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    chatbot = pipeline("text-classification", model=model, tokenizer=tokenizer)

    label_to_id = {v: k for k, v in model.config.label2id.items()}
    chatbot_instance = SimpleChatbot(label_to_id, intents_data)

    chatbot_instance.interact(chatbot)
