import argparse

from src.runner import ModelRunner

from src.utils.utils import (LOGGER,)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="configs/model.yaml", help='model path(s)')
    parser.add_argument('--data', type=str, default="configs/val.yaml", help='model path(s)')
    parser.add_argument('--data-name', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default="cpu", help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    opt = parser.parse_args()
    LOGGER.info(f"Running validation script: {vars(opt)}")
    return opt


if __name__ == "__main__":

    opt = parse_opt()
    runner = ModelRunner(opt)
    runner.validate(**vars(opt))
