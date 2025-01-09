from model.model_trainer import ModelTrainer
from transformers import BertForSequenceClassification, BertTokenizerFast


def initialize_model(num_labels, id_to_label, label_to_id):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_to_id,
    )
    return model


def train_model(model, train_dataset, test_dataset, training_args, metrics_calculator):
    model_trainer = ModelTrainer(model, train_dataset, test_dataset, training_args)
    model_trainer.train(metrics_calculator.compute)
    return model_trainer


def save_and_load_model(model_trainer, model_path, tokenizer):
    model_trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    return model, tokenizer
