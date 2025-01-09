#  BERT-based virtual assistant for student queries

## Overview

This project uses the BERT-based model for sequence classification to build a chatbot that can understand and respond to user queries about LJ University. The chatbot is fine-tuned with a custom dataset derived from the intents.json file, containing user input patterns and associated intent labels (tags). The `bert-base-uncased` pre-trained model from Hugging Face is used, providing a robust foundation for natural language understanding tasks. The model is trained to classify user input into specific intent categories, ensuring that the chatbot provides contextually relevant responses to questions about university admission, course details, and more. The project incorporates tokenization, model training, and evaluation processes, along with chatbot interaction handling.

## Usage

Navigate to the `/chatbot/api` directory and run the following command to start the server:

```
uvicorn chatbot_api:app --reload

```
Once the server is running, open the `index.html` file located in the `/api/gui` directory to access the chatbot interface.

![](screenshot.jpg)

## Requirements

- numpy==1.26.4
- pandas==2.2.2
- sklearn==1.6.0
- torch==2.5.1+cu121
- transformers==4.47.1
- FastAPI==0.115.6
- Pydantic==2.10.3
- Uvicorn==0.32.1


