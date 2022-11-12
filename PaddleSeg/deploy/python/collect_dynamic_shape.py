# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import codecs
import os
import sys

import yaml
import numpy as np
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..', '..'))

from paddleseg.utils import logger, get_image_list, progbar
from infer import DeployConfig
"""
Load images and run the model, it collects and saves dynamic shapes,
which are used in deployment with TRT.
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        "--config",
        help="The deploy config generated by exporting model.",
        type=str,
        required=True)
    parser.add_argument(
        '--image_path',
        help='The directory or path or file list of the images to be predicted.',
        type=str,
        required=True)

    parser.add_argument(
        '--dynamic_shape_path',
        type=str,
        default="./dynamic_shape.pbtxt",
        help='The path to save dynamic shape.')

    return parser.parse_args()


def is_support_collecting():
    return hasattr(PredictConfig, "collect_shape_range_info") \
        and hasattr(PredictConfig, "enable_tuned_tensorrt_dynamic_shape")


def collect_dynamic_shape(args):

    if not is_support_collecting():
        logger.error("The Paddle does not support collecting dynamic shape, " \
            "please reinstall the PaddlePaddle (latest gpu version).")

    # prepare config
    cfg = DeployConfig(args.config)
    pred_cfg = PredictConfig(cfg.model, cfg.params)
    pred_cfg.enable_use_gpu(1000, 0)
    pred_cfg.collect_shape_range_info(args.dynamic_shape_path)

    # create predictor
    predictor = create_predictor(pred_cfg)
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    # get images
    img_path_list, _ = get_image_list(args.image_path)
    if not isinstance(img_path_list, (list, tuple)):
        img_path_list = [img_path_list]
    logger.info(f"The num of images is {len(img_path_list)} \n")

    # collect
    progbar_val = progbar.Progbar(target=len(img_path_list))
    for idx, img_path in enumerate(img_path_list):
        data = np.array([cfg.transforms(img_path)[0]])
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data)

        try:
            predictor.run()
        except:
            logger.info(
                "Fail to collect dynamic shape. Usually, the error is out of "
                "GPU memory, for the model and image are too large.\n")
            del predictor
            if os.path.exists(args.dynamic_shape_path):
                os.remove(args.dynamic_shape_path)

        progbar_val.update(idx + 1)

    logger.info(f"The dynamic shape is save in {args.dynamic_shape_path}")


if __name__ == '__main__':
    args = parse_args()
    collect_dynamic_shape(args)
