class TextMultiLabelDataset(torchtext.data.Dataset):
    def __init__(self, df, tt_text_field, tt_label_field, txt_col, lbl_cols, **kwargs):
        # torchtext Field objects
        fields = [('text', tt_text_field)]
        for l in lbl_cols: fields.append((l, tt_label_field))
            
        is_test = False if lbl_cols[0] in df.columns else True
        n_labels = len(lbl_cols)
        
        examples = []
        for idx, row in df.iterrows():
            if not is_test:
                lbls = [ row[l] for l in lbl_cols ]
            else:
                lbls = [0.0] * n_labels
                
            txt = str(row[txt_col])
            examples.append(data.Example.fromlist([txt]+lbls, fields))
                            
        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(example): 
        return len(example.text)
    
    @classmethod
    def splits(cls, text_field, label_field, train_df, txt_col, lbl_cols, val_df=None, test_df=None, **kwargs):
        # build train, val, and test data
        train_data, val_data, test_data = (None, None, None)
        
        if train_df is not None: 
            train_data = cls(train_df.copy(), text_field, label_field, txt_col, lbl_cols, **kwargs)
        if val_df is not None: 
            val_data = cls(val_df.copy(), text_field, label_field, txt_col, lbl_cols, **kwargs)
        if test_df is not None: 
            test_data = cls(test_df.copy(), text_field, label_field, txt_col, lbl_cols, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)
    
    
class TextMultiLabelDataLoader():
    def __init__(self, src, x_fld, y_flds, y_dtype='torch.cuda.FloatTensor'):
        self.src, self.x_fld, self.y_flds = src, x_fld, y_flds
        self.y_dtype = y_dtype

    def __len__(self): return len(self.src)#-1

    def __iter__(self):
        it = iter(self.src)
        for i in range(len(self)):
            b = next(it)
            
            if (len(self.y_flds) > 1):
                targ = [ getattr(b, y) for y in self.y_flds ] 
                targ = torch.stack(targ, dim=1).type(self.y_dtype)
            else: 
                targ = getattr(b, self.y_flds[0])
                targ = targ.type(self.y_dtype)

            yield getattr(b, self.x_fld), targ

            
class TextMultiLabelData(ModelData):

    @classmethod
    def from_splits(cls, path, splits, bs, text_name='text', label_names=['label'], 
                    target_dtype='torch.cuda.FloatTensor'):
        
        text_fld = splits[0].fields[text_name]
        
        label_flds = []
        if (len(label_names) == 1): 
            label_fld = splits[0].fields[label_names[0]]
            label_flds.append(label_fld)
            if (label_fld.use_vocab): 
                label_fld.build_vocab(splits[0])
                target_dtype = 'torch.cuda.LongTensor'
        else:
            for n in label_names:
                label_fld = splits[0].fields[n]
                label_flds.append(label_fld)

        iters = torchtext.data.BucketIterator.splits(splits, batch_size=bs)
        trn_iter,val_iter,test_iter = iters[0],iters[1],None
        test_dl = None
        if len(iters) == 3:
            test_iter = iters[2]
            test_dl = TextMultiLabelDataLoader(test_iter, text_name, label_names, target_dtype)
        trn_dl = TextMultiLabelDataLoader(trn_iter, text_name, label_names, target_dtype)
        val_dl = TextMultiLabelDataLoader(val_iter, text_name, label_names, target_dtype)

        obj = cls.from_dls(path, trn_dl, val_dl, test_dl)
        obj.bs = bs
        obj.pad_idx = text_fld.vocab.stoi[text_fld.pad_token]
        obj.nt = len(text_fld.vocab)

        # if multiple labels, assume the # of classes = the # of labels 
        if (len(label_names) > 1):
            c = len(label_names)
        # if label has a vocab, assume the vocab represents the # of classes
        elif (hasattr(label_flds[0], 'vocab')): 
            c = len(label_flds[0].vocab)
        else:
            c = 1
            
        obj.c = c

        return obj