import sys
import ax
from ax.service.managed_loop import optimize
sys.path.append(".")
from train.train_normal import train_eval
from models.mlp2gcd import two_layer_mlp_dropout, two_layer_mlp, LR
from tasks.utils import yaml2dict, load_gcd, load_gcd_train_val
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

def main(train_cfg):
    cfg = yaml2dict(train_cfg)
    net = two_layer_mlp(59, cfg.get("num_classes"))
    train_loader = load_gcd(cfg)
    test_loader = load_gcd(cfg, 'val')
    def bo_eval_func(params):
        train_loader_bo, val_loader_bo = load_gcd_train_val(cfg)
        net_bo = two_layer_mlp(59, cfg.get("num_classes"))
        score = train_eval(net_bo, train_loader_bo, val_loader_bo, params, dtype, device)
        return score
    parameters = [{"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
                  {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
                  {"name": "weight_decay", "type": "range", "bounds": [0.0, 0.01]}]
    best_parameters, values, experiment, model = optimize(parameters,
                                                          evaluation_function=bo_eval_func,
                                                          objective_name="accuracy")
    print(best_parameters)
    acc = train_eval(net, train_loader, test_loader, best_parameters, dtype, device)
    print("Final accuracy is {}".format(acc))

train_cfg = '../cfg/gcd_mlp.yaml'
main(train_cfg)

