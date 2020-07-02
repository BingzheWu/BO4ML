import yaml
from dataset.gcd import gcd
from torch.utils.data import DataLoader, random_split

def yaml2dict(cfg_file):
    with open(cfg_file, 'r') as f:
        cfg = yaml.load(f)
    return cfg

def load_gcd(cfg, mode='train'):
    dataset = gcd(cfg, mode)
    batch_size = cfg.get('batch_size')
    if mode=='train':
        shuffle = True
    else:
        shuffle = False
    num_workers = cfg.get('num_workers', 4)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader

def load_gcd_train_val(cfg, mode='train'):
    dataset = gcd(cfg, mode)
    train_set, val_set = random_split(dataset, [400, 113])
    batch_size = cfg.get('batch_size')
    num_workers = cfg.get('num_workers', 4)
    train_loader = DataLoader(train_set,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers)
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    return train_loader, val_loader