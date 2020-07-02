import sys
import ax
from ax.service.managed_loop import optimize
sys.path.append(".")
from train.train_normal import train_eval
from models.mlp2gcd import two_layer_mlp_dropout, two_layer_mlp, LR
from tasks.utils import yaml2dict, load_gcd
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

def main(train_cfg):
    cfg = yaml2dict(train_cfg)
    net = two_layer_mlp(59, cfg.get("num_classes"))
    train_loader = load_gcd(cfg)
    test_loader = load_gcd(cfg, 'val')
    acc = train_eval(net, train_loader, test_loader, cfg, dtype, device)
    print("Final accuracy is {}".format(acc))

train_cfg = '../cfg/gcd_mlp.yaml'
main(train_cfg)

