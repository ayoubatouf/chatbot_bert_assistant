from chatbot.simple_chatbot import SimpleChatbot


def start_chat(chatbot, label_to_id, intents_data):
    chatbot_instance = SimpleChatbot(label_to_id, intents_data)
    chatbot_instance.interact(chatbot)
