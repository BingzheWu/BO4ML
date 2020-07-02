import sys
import ax
from ax.service.managed_loop import optimize
sys.path.append(".")
from train.train_adam import train_eval
from models.LeNet5 import LeNet5, load_mnist
from tasks.utils import yaml2dict
import torch
dtype = torch.float
#torch.distributed.init_process_group(backend="nccl")
#local_rank = torch.distributed.get_rank()
#torch.cuda.set_device(local_rank)
print(torch.cuda.is_available())
device = torch.device("cuda")

def main(train_cfg):
    cfg = yaml2dict(train_cfg)
    net = LeNet5()
    '''
    if torch.cuda.device_count() > 1:
        net = torch.nn.parallel.DataParallel(net)
    '''
    train_loader, val_loader, test_loader = load_mnist(batch_size=128, num_workers=64)
    combined_train_valid_set = torch.utils.data.ConcatDataset([
        train_loader.dataset.dataset,
        val_loader.dataset.dataset,
    ])
    combined_train_valid_loader = torch.utils.data.DataLoader(
        combined_train_valid_set,
        batch_size=512,
        shuffle=True,
        num_workers=128
    )
    acc = train_eval(net, combined_train_valid_loader, test_loader, cfg, dtype, device)
    print("Final accuracy is {}".format(acc))

train_cfg = '../cfg/bo_mnist.yaml'
main(train_cfg)

