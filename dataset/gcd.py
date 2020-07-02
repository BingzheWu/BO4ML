
import pandas as pd
import torch.utils.data as data
import yaml
import numpy as np

class gcd(data.Dataset):
    def __init__(self, cfg, mode='train'):
        super(gcd, self).__init__()
        self.train_csv = cfg.get("train_csv")
        self.test_csv = cfg.get("test_csv")
        if mode == 'train':
            self.df = pd.read_csv(self.train_csv)
        elif mode == 'val':
            self.df = pd.read_csv(self.test_csv)

    def __getitem__(self, index):
        feature = self.df.iloc[index,0:-3]
        label = self.df.iloc[index, -3:-1]
        label = np.array(label)
        label = int(np.argmax(label))
        return np.array(feature).astype(np.float32), label
    def __len__(self):
        return self.df.shape[0]

def _test_gcd():
    with open("cfg/gcd_mlp.yaml") as f:
        cfg = yaml.load(f)
    data = gcd(cfg)
    feature, label = data.__getitem__(0)
    print(feature.dtype)


if __name__ == '__main__':
    _test_gcd()