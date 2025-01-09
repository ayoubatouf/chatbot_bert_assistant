class DataLoaderInterface:
    def load_data(self, filename):
        raise NotImplementedError

    def prepare_data(self):
        raise NotImplementedError
