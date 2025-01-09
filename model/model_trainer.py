from transformers import Trainer


class ModelTrainer:
    def __init__(self, model, train_dataloader, test_dataloader, training_args):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.training_args = training_args
        self.trainer = None

    def train(self, compute_metrics):
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataloader,
            eval_dataset=self.test_dataloader,
            compute_metrics=compute_metrics,
        )
        self.trainer.train()

    def save_model(self, path):
        self.trainer.save_model(path)
