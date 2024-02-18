#python image_demo.py /workspace/FishEye8k/dataset/Fisheye8K_all_including_train/test/images/camera1_A_11.png configs/coco/cascade_internimage_xl_fpn_3x_coco.py checkpoints/cascade_internimage_xl_fpn_3x_coco.pth

# modifiey from https://github.com/OpenGVLab/InternImage/blob/master/detection/image_demo.py
import requests, mmcv, asyncio, argparse
# F401: indicates that a module imported in the code is not useda
# F403: indicates that a particular import is shadowed by another import
import mmcv_custom   # noqa: F401,F403
import mmdet_custom  # noqa: F401,F403
from pathlib import Path
from mmdet.apis import (async_inference_detector, inference_detector, init_detector, show_result_pyplot)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    config = "configs/cascade_internimage_xl_fpn_3x_coco.py"
    checkpoint = "checkpoints/cascade_internimage_xl_fpn_3x_coco.pth"
    if not Path(checkpoint).exists():
        print("Model checkpoint doesn't exist. Downloading...")
        # ideally to use tqdm to show progress
        response = requests.get("https://huggingface.co/OpenGVLab/InternImage/resolve/main/cascade_internimage_xl_fpn_3x_coco.pth")
        Path(checkpoint).write_bytes(response.content)
    # build the model from a config file and a checkpoint file
    model = init_detector(config, checkpoint, device="cuda:0")
    # test a single image
    result = inference_detector(model, args.img)
    
    mmcv.mkdir_or_exist(args.out)
    out_file = Path(args.out) / Path(args.img).name
    # show the results
    model.show_result(
        args.img,
        result,
        score_thr=args.score_thr,
        show=False,
        bbox_color=args.palette,
        text_color=(200, 200, 200),
        mask_color=args.palette,
        out_file=out_file
    )
    print(f"Result is save at {out_file}")



if __name__ == '__main__':
    args = parse_args()
    main(args)
