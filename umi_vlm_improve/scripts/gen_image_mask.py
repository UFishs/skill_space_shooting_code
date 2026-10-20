# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import imageio
import pickle
import json
import numpy as np
from umi.common.cv_util import get_gripper_with_finger_mask, get_mirror_mask

# %%
@click.command()
@click.option('-c', '--config', required=False, default='umi/asset/mask.json', help='Mask config path')
@click.option('-o', '--output', required=True, help='Output png path')
@click.option('-h', '--height', required=False, default=2028, help='Image height')
@click.option('-w', '--width', required=False, default=2704, help='Image width')
def main(config, output, height, width):
    json_data = json.load(open(config, 'r'))
    mask_image = np.zeros([height, width, 3], dtype=np.uint8)
    mask_image = get_mirror_mask(json_data, mask_image, (255, 255, 255))
    mask_image = get_gripper_with_finger_mask(mask_image, color=(255, 255, 255))
    
    imageio.imwrite(output, mask_image)

    # # load gripper image
# %%
if __name__ == '__main__':
    main()
