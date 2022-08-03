from __future__ import print_function
import os
import torch
import train_options
from train_Processor import Processor


def main():
    args = train_options.parser.parse_args()
    # assert train
    assert args.run_type in [0, 1, 2]
    # fix random seed for stable results
    torch.manual_seed(args.seed)
    # set visible gpus
    args.gpu_ids = visible_gpu(args.gpus)
    # create folder
    if not os.path.exists('./ckpt/'):
        os.makedirs('./ckpt/')

    out_dir = os.path.join('./ckpt/', str(args.lang), str(args.model_id))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    processor = Processor(args)
    processor.processing()


def visible_gpu(gpus):
    """
        set visible gpu.
        can be a single id, or a list
        return a list of new gpus ids
    """
    gpus = [gpus] if isinstance(gpus, int) else list(gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(map(str, gpus)))
    return list(range(len(gpus)))


if __name__ == '__main__':
    main()
