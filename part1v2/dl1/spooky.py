import torchtext
from torchtext import data


class SpookyDataset(torchtext.data.Dataset):
    def __init__(self, df, text_field, label_field, **kwargs):
        fields = [('text', text_field), ('label', label_field)]

        examples = [ data.Example.fromlist([row['text'], row['author'] if 'author' in row else None], fields) for index, row in df.iterrows() ]
        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(example): return len(example.text)
    
    @classmethod
    def splits(cls, text_field, label_field, train_df, val_df=None, test_df=None, **kwargs):
        # build train, val, and test data
        train_data = None if train_df is None else cls(train_df, text_field, label_field, **kwargs)
        val_data = None if val_df is None else cls(val_df, text_field, label_field, **kwargs)
        test_data = None if test_df is None else cls(test_df, text_field, label_field, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)
