import argparse

from src.runner import ModelRunner


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-type', default='train', choices=['train', 'val',], help='type of job to run, train, val, etc')
    parser.add_argument('--resume', action='store_true', help='only save final checkpoint')
    parser.add_argument('--model', type=str, default="configs/model.yaml", help='initial weights path')
    parser.add_argument('--data', type=str, default="configs/data.yaml", help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=50, help='total training epochs')
    parser.add_argument('--no-save', action='store_true', help='only save final checkpoint')
    parser.add_argument('--no-val', action='store_true', help='only validate final epoch')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--optimizer', type=str, default="configs/opt.yaml", help='dataset.yaml path')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default="pytorch-cifar10", help='save to project/name')
    parser.add_argument('--save-period', type=int, default=float("inf"), help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--bin-dir', type=str, default="bin")
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    runner = ModelRunner(opt)
    if opt.run_type == "train":
        runner.train()
    elif opt.run_type == "val":
        ModelRunner.validate(model=opt.model, data="configs/val_data.yaml")
