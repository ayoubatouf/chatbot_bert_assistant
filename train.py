from sklearn.model_selection import train_test_split
from helper.data import create_datasets, load_and_prepare_data, tokenize_data
from helper.model import initialize_model, save_and_load_model, train_model
from helper.utils import load_json_data, set_random_seed
from transformers import BertTokenizer
from transformers import TrainingArguments
from model.metrics_calculator import MetricsCalculator


if __name__ == "__main__":
    set_random_seed(seed=100)
    filename = "intents.json"
    data_loader = load_and_prepare_data(filename)
    intents_data = load_json_data(filename)

    patterns = list(data_loader.df["Pattern"])
    labels = list(data_loader.df["label_ids"])
    X_train, X_test, y_train, y_test = train_test_split(
        patterns, labels, random_state=100
    )

    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    train_encodings, test_encodings = tokenize_data(X_train, X_test, tokenizer)

    train_dataset, test_dataset = create_datasets(
        train_encodings, test_encodings, y_train, y_test
    )

    num_labels = len(data_loader.unique_labels)
    model = initialize_model(
        num_labels, data_loader.id_to_label, data_loader.label_to_id
    )

    training_args = TrainingArguments(
        output_dir="./output",
        run_name="unique_run_name",
        do_train=True,
        do_eval=True,
        num_train_epochs=100,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.05,
        logging_strategy="steps",
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",
    )

    metrics_calculator = MetricsCalculator()

    model_trainer = train_model(
        model, train_dataset, test_dataset, training_args, metrics_calculator
    )

    model_path = "pretrained_chatbot_model"
    save_and_load_model(model_trainer, model_path, tokenizer)
