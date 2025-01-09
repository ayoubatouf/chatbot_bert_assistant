from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  
from pydantic import BaseModel
from transformers import pipeline, BertTokenizerFast, BertForSequenceClassification
import random
from helper.utils import load_json_data

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)


class UserInput(BaseModel):
    text: str


def load_resources(model_path = "../pretrained_chatbot_model", intents_path = "../intents.json"):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    
    chatbot = pipeline("text-classification", model=model, tokenizer=tokenizer)
    intents_data = load_json_data(intents_path)
    
    return chatbot, intents_data

chatbot, intents_data = load_resources()


def get_chatbot_response(user_input_text):
    response = chatbot(user_input_text)[0]
    label = response.get('label')
    
    if not label:
        return "Sorry, I can't recognize your intent."
    
    try:
        for intent in intents_data['intents']:
            if intent['tag'].lower() == label:
                return random.choice(intent['responses'])
    except KeyError:
        return "Sorry, I don't know how to respond to that."

    return "Sorry, I couldn't find an appropriate response."

@app.post("/chat/")
def chat_with_bot(user_input: UserInput):
    user_input_text = user_input.text
    bot_response = get_chatbot_response(user_input_text)
    return {"response": bot_response}
