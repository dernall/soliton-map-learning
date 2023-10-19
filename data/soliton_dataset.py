import numpy as np
import torch
from torch.utils.data import Dataset

class SolitonDataset(Dataset):

    def __init__(self, df, to_db=False, train=False, scaler=None):
        super().__init__()
        self.labels = df["target"]
        self.spectrums = df.drop(columns=["target"])
        self.index = df.index.values
        keys = set(self.labels)
        self.dict_id = dict(zip(keys, range(len(keys))))

        if to_db:
            self.spectrums = self._to_db(self.spectrums)
        if scaler:
            if train:
                self.spectrums = scaler.fit_transform(self.spectrums)
            else:
                self.spectrums = scaler.transform(self.spectrums)
        

    def _to_db(x):
        return 10 * np.log10(x)

    def __len__(self):
        return len(self.spectrums)

    def __getitem__(self, index):
        if not isinstance(self.spectrums, np.ndarray):
          self.spectrums = np.array(self.spectrums)
        anc = self.spectrums[index]

        anchor_label = self.labels.iloc[index]

        positive_list = self.index[self.index!=index][self.labels[self.index!=index]==anchor_label]
        positive_item = np.random.choice(positive_list)
        pos = self.spectrums[positive_item]

        negative_list = self.index[self.index != index][self.labels[self.index != index] != anchor_label]
        negative_item = np.random.choice(negative_list)
        neg = self.spectrums[negative_item]

        return torch.tensor(anc).type('torch.FloatTensor'), torch.tensor(pos).type('torch.FloatTensor'), torch.tensor(neg).type('torch.FloatTensor'), torch.tensor(self.dict_id[anchor_label])