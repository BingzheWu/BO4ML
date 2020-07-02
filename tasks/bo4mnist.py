import sys
import ax
from ax.service.managed_loop import optimize
sys.path.append(".")
from train.train_adam import train_eval
from models.LeNet5 import LeNet5, load_mnist
from tasks.utils import yaml2dict
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

def main(train_cfg):
    cfg = yaml2dict(train_cfg)
    net = LeNet5()
    train_loader = load_mnist(batch_size=128)
    train_loader, val_loader, test_loader = load_mnist(batch_size=128, num_workers=64)
    def bo_eval_func(params):
        net_bo = LeNet5()
        score = train_eval(net_bo, train_loader, val_loader, params, dtype, device)
        return score
    parameters = [{"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
                  {"name": "beta1", "type": "range", "bounds": [0.0, 1.0]},
                  {"name": "beta2", "type": "range", "bounds": [0.0, 1.0]},
                  {"name": "weight_decay", "type": "range", "bounds": [0.0, 0.01]}]
    best_parameters, values, experiment, model = optimize(parameters,
                                                          evaluation_function=bo_eval_func,
                                                          objective_name="accuracy")
    print(best_parameters)
    combined_train_valid_set = torch.utils.data.ConcatDataset([
        train_loader.dataset.dataset,
        val_loader.dataset.dataset,
    ])
    combined_train_valid_loader = torch.utils.data.DataLoader(
        combined_train_valid_set,
        batch_size=128,
        shuffle=True,
        num_workers=64
    )
    acc = train_eval(net, combined_train_valid_loader, test_loader, best_parameters, dtype, device)
    print("Final accuracy is {}".format(acc))

train_cfg = '../cfg/bo_mnist.yaml'
main(train_cfg)

