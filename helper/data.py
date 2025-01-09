from data.intent_data_loader import IntentDataLoader
from data.tokenized_dataset import TokenizedDataset
from helper.utils import load_json_data


def load_and_prepare_data(filename):
    intents_data = load_json_data(filename)
    data_loader = IntentDataLoader(intents_data)
    data_loader.load_data()
    return data_loader


def tokenize_data(train_texts, test_texts, tokenizer, max_length=256):
    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=max_length
    )
    test_encodings = tokenizer(
        test_texts, truncation=True, padding=True, max_length=max_length
    )
    return train_encodings, test_encodings


def create_datasets(train_encodings, test_encodings, train_labels, test_labels):
    train_dataset = TokenizedDataset(train_encodings, train_labels)
    test_dataset = TokenizedDataset(test_encodings, test_labels)
    return train_dataset, test_dataset
