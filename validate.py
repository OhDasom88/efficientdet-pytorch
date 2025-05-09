#!/usr/bin/env python
""" COCO validation script

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import time
import torch
import torch.nn.parallel
from contextlib import suppress

from effdet import create_model, create_evaluator, create_dataset, create_loader
from effdet.data import resolve_input_config
from timm.utils import AverageMeter, setup_default_logging
try:
    from timm.layers import set_layer_config
except ImportError:
    from timm.models.layers import set_layer_config

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True


def add_bool_arg(parser, name, default=False, help=''):  # FIXME move to utils
    dest_name = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=dest_name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=dest_name, action='store_false', help=help)
    parser.set_defaults(**{dest_name: default})


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument(
    # 'root', # 2025-05-08 dasom
    '--root', # 2025-05-08 dasom
    metavar='DIR',
    # default='VOCdevkit', # 2025-05-08 dasom
    default='/home/sysnova/pnid/doosan/efficientdet-pytorch/VOCdevkit', # 2025-05-08 dasom
    help='path to dataset root')
parser.add_argument('--dataset', 
                    # default='coco', # 2025-05-08 dasom
                    # default='voc2007', # 2025-05-08 dasom
                    default='PNID250508v1', # 2025-05-08 dasom
                    type=str, metavar='DATASET',
                    help='Name of dataset (default: "coco"')
parser.add_argument('--split', default='val',
                    help='validation split')
parser.add_argument('--model', '-m', metavar='MODEL', 
                    # default='tf_efficientdet_d1', # 2025-05-08 dasom
                    # default='efficientdet_d0', # 2025-05-08 dasom
                    default='tf_efficientdet_d7x', # 2025-05-09 dasom
                    help='model architecture (default: tf_efficientdet_d1)')
add_bool_arg(parser, 'redundant-bias', default=None,
                    help='override model config for redundant bias layers')
add_bool_arg(parser, 'soft-nms', default=None, help='override model config for soft-nms')
parser.add_argument('--num-classes', type=int, 
                    # default=None, # 2025-05-08 dasom
                    # default=20, # 2025-05-08 dasom
                    default=7, # 2025-05-08 dasom
                    metavar='N',
                    help='Override num_classes in model config if set. For fine-tuning from pretrained.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', 
                    # default=128, 
                    default=2, # 2025-05-08 dasom
                    # default=64, # 2025-05-09 dasom
                    type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='bilinear', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--fill-color', default=None, type=str, metavar='NAME',
                    help='Image augmentation fill (background) color ("mean" or int)')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', 
                    # default='', # 2025-05-08 dasom
                    # default='/home/sysnova/pnid/doosan/efficientdet-pytorch/output/train/20250507-133937-efficientdet_d0/model_best.pth.tar', 
                    # default='/home/sysnova/pnid/doosan/efficientdet-pytorch/output/train/20250508-042937-efficientdet_d0/model_best.pth.tar', # 2025-05-08 dasom
                    # default='/home/sysnova/pnid/doosan/efficientdet-pytorch/output/train/20250508-112033-efficientdet_d0/model_best.pth.tar', # 2025-05-08 dasom
                    # default='/home/sysnova/pnid/doosan/efficientdet-pytorch/output/train/20250508-112033-efficientdet_d0/last.pth.tar', # 2025-05-08 dasom
                    default='/home/sysnova/pnid/doosan/efficientdet-pytorch/output/train/20250508-123728-tf_efficientdet_d7x/model_best.pth.tar', # 2025-05-09 dasom
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                    help="Enable compilation w/ specified backend (default: inductor).")
parser.add_argument('--results', default='', type=str, metavar='FILENAME',
                    help='JSON filename for evaluation results')


def validate(args):
    setup_default_logging()

    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    args.pretrained = args.pretrained or not args.checkpoint  # might as well try to validate something
    args.prefetcher = not args.no_prefetcher

    # create model
    with set_layer_config(scriptable=args.torchscript):
        extra_args = {}
        if args.img_size is not None:
            extra_args = dict(image_size=(args.img_size, args.img_size))
        bench = create_model(
            args.model,
            bench_task='predict',
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            redundant_bias=args.redundant_bias,
            soft_nms=args.soft_nms,
            checkpoint_path=args.checkpoint,
            checkpoint_ema=args.use_ema,
            **extra_args,
        )
    model_config = bench.config

    param_count = sum([m.numel() for m in bench.parameters()])
    print('Model %s created, param count: %d' % (args.model, param_count))

    bench = bench.cuda()
    # device = torch.device('cuda:1')
    # bench.to(device)

    if args.torchscript:
        assert not args.apex_amp, \
            'Cannot use APEX AMP with torchscripted model, force native amp with `--native-amp` flag'
        bench = torch.jit.script(bench)
    elif args.torchcompile:
        bench = torch.compile(bench, backend=args.torchcompile)

    amp_autocast = suppress
    if args.apex_amp:
        bench = amp.initialize(bench, opt_level='O1')
        print('Using NVIDIA APEX AMP. Validating in mixed precision.')
    elif args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        print('Using native Torch AMP. Validating in mixed precision.')
    else:
        print('AMP not enabled. Validating in float32.')

    if args.num_gpu > 1:
        bench = torch.nn.DataParallel(bench, device_ids=list(range(args.num_gpu)))

    dataset = create_dataset(args.dataset, args.root, args.split)
    input_config = resolve_input_config(args, model_config)
    loader = create_loader(
        dataset,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=args.workers,
        pin_mem=args.pin_mem,
        # device=device, # 2025-05-09 dasom 학습할때 0번 GPU를ㅋ 쓰고 있어서 추론할때는 1번 GPU를 써야함
    )

    evaluator = create_evaluator(args.dataset, dataset, pred_yxyx=False)
    bench.eval()
    batch_time = AverageMeter()
    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            # with amp_autocast(): # 2025-05-08 dasom
            with torch.amp.autocast(device_type='cuda'): # 2025-05-08 dasom
            # with torch.amp.autocast(device_type=device, dtype=torch.bfloat16): # 2025-05-09 dasom
                # 2025-05-08 dasom
                # output.shape = batch_size, 100, 6
                # 6 = (x1, y1, x2, y2, score, class).length
                # class = 1 ~ 20
                output = bench(input, img_info=target)

                
            evaluator.add_predictions(output, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.log_freq == 0 or i == last_idx:
                print(
                    f'Test: [{i:>4d}/{len(loader)}]  '
                    f'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {input.size(0) / batch_time.avg:>7.2f}/s)  '
                )

        # 2025-05-08 dasom
        # 예측결과 시각화
        # 정답은 초록, 예측은 빨강
        # parser = dataset.parser
        from PIL import Image, ImageDraw
        import numpy as np
        cat_id_to_label = {v: k for k, v in dataset.parser.cat_id_to_label.items()}
        for img_idx, preds, inp in zip(
            target.get('img_idx').tolist(), 
            output.tolist(),
            input.tolist()
            ):
            img_info = dataset.parser.img_infos[img_idx]
            img_path = dataset.data_dir / img_info.get('file_name')
            img = Image.open(img_path).convert('RGB')
            img_copy = img.copy()
            draw = ImageDraw.Draw(img_copy)
            ann = dataset.parser.get_ann_info(img_idx)
            bboxes = ann.get('bbox')
            clses = ann.get('cls')
            from tqdm import tqdm
            for bbox, cls in tqdm(zip(bboxes.tolist(), clses.tolist()), desc='정답'):
                y1, x1, y2, x2 = bbox # 왜 y1, x1, y2, x2 순서인지 모르겠음, 2025-05-08 dasom
                draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
                draw.text((x1, y1), str(cat_id_to_label[cls]), fill='green')
            # img_copy.save(f'gt.png')

            # input_img = np.array(inp).astype(np.uint8).transpose(1,2,0)
            # img = Image.fromarray(input_img)

            # img_copy = img.copy()
            # draw = ImageDraw.Draw(img_copy)
            for x1, y1, x2, y2, score, cls in tqdm(preds, desc='예측'):
            # for y1, x1, y2, x2, score, cls in tqdm(preds, desc='예측'):
                # if score < 0.2:
                #     continue
                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                draw.text((x1, y1), f'{str(cat_id_to_label[cls])} {score:.2f}', fill='red')

            img_copy.save(f'pred.png')
            break
            # 여기까지 마지막 배치의 예측결과 시각화 2025-05-08 dasom

    mean_ap = 0.
    if dataset.parser.has_labels:
        mean_ap = evaluator.evaluate(output_result_file=args.results)
    else:
        evaluator.save(args.results)

    return mean_ap


def main():
    args = parser.parse_args()
    validate(args)


if __name__ == '__main__':
    main()

