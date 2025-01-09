from chatbot.chatbot_inteface import ChatbotInterface
import random


class SimpleChatbot(ChatbotInterface):
    def __init__(self, label_to_id, intents):
        self.label_to_id = label_to_id
        self.intents = intents

    def interact(self, chatbot):
        print("Hi! I'm your virtual assistant. Feel free to ask questions")
        print("Type 'quit' to exit the chat\n\n")
        user_input = input("User: ").strip().lower()

        while user_input != "quit":
            response = chatbot(user_input)[0]
            print(f"DEBUG: Model response: {response}")
            label = response.get("label")

            if not label:
                print("Chatbot: Sorry, I can't recognize your intent.\n\n")
            else:
                try:
                    intent_responses = None
                    for intent in self.intents["intents"]:
                        if intent["tag"].lower() == label:
                            intent_responses = intent["responses"]
                            break

                    if intent_responses:
                        response_text = random.choice(intent_responses)
                        print(f"CHATBOT: {response_text}\n\n")
                    else:
                        print(
                            f"Chatbot: Sorry, I don't know how to respond to '{label}'.\n\n"
                        )
                except KeyError:
                    print(
                        f"Chatbot: Sorry, I don't know how to respond to '{label}'.\n\n"
                    )

            user_input = input("User: ").strip().lower()
