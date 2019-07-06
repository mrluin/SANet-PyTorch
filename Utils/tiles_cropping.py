import cv2
import tqdm
import os
import glob
from Utils.tools import ensure_dir
from PIL import Image
from Configs.config import Configurations
import argparse

def image_cropper(configs, subset):
    """
    :param config: config include:
    the size of tiles,
    the path to source directory(image and label),
    the path to save_directory
    """

    # ensure output directory
    ensure_dir(os.path.join(configs.path_cropped_images, subset))
    ensure_dir(os.path.join(configs.path_cropped_labels, subset))

    images_source = glob.glob(os.path.join(configs.path_source_images, subset, '*.tif'))
    labels_source = glob.glob(os.path.join(configs.path_source_labels, subset, '*.tif'))

    size_original_h = configs.size_original_h
    size_original_w = configs.size_original_w
    size_cropped_h = configs.size_cropped_images_h
    size_cropped_w = configs.size_cropped_images_w
    size_overlap = configs.size_overlap

    stride_h = size_cropped_h - size_overlap
    stride_w = size_cropped_w - size_overlap
    # per row and column
    #N = size_original / size_cropped + 1 if size_original % size_cropped != 0 else size_original / size_cropped

    for path_image, path_label in tqdm.tqdm(zip(images_source, labels_source)):
        count = 0
        image_name = path_image.split('/')[-1].split('.')[0]
        label_name = path_label.split('/')[-1].split('.')[0]
        image = cv2.imread(path_image)
        label = cv2.imread(path_label)

        # exclude the last column and last row
        for h_start in range(0, size_original_h-stride_h, stride_h):
            for w_start in range(0, size_original_w-stride_w, stride_w):
                tile = image[h_start:h_start+size_cropped_h, w_start:w_start+size_cropped_w, :]
                tile_name = image_name + '_{}.tif'.format(count)

                tile_label = label[h_start:h_start+size_cropped_h, w_start:w_start+size_cropped_w, :]
                tile_label_name = label_name + '_{}.tif'.format(count)

                path_save_t = os.path.join(configs.path_cropped_images, subset, tile_name)
                path_save_l = os.path.join(configs.path_cropped_labels, subset, tile_label_name)

                cv2.imwrite(path_save_t, tile)
                cv2.imwrite(path_save_l, tile_label)
                count += 1
            # for the last column per row
            tile = image[h_start: h_start+size_cropped_h, size_original_w-size_cropped_w:size_original_w, :]
            tile_name = image_name + '_{}.tif'.format(count)

            tile_label = label[h_start:h_start + size_cropped_h, size_original_w-size_cropped_w:size_original_w, :]
            tile_label_name = label_name + '_{}.tif'.format(count)

            path_save_t = os.path.join(configs.path_cropped_images, subset, tile_name)
            path_save_l = os.path.join(configs.path_cropped_labels, subset, tile_label_name)

            cv2.imwrite(path_save_t, tile)
            cv2.imwrite(path_save_l, tile_label)
            count += 1

        # for the last row
        for w_start in range(0, size_original_w-stride_w, stride_w):
            tile = image[size_original_h-size_cropped_h:size_original_h, w_start:w_start+size_cropped_w, :]
            tile_name = image_name + '_{}.tif'.format(count)

            tile_label = label[size_original_h-size_cropped_h:size_original_h, w_start:w_start+size_cropped_w, :]
            tile_label_name = label_name + '_{}.tif'.format(count)

            path_save_t = os.path.join(configs.path_cropped_images, subset, tile_name)
            path_save_l = os.path.join(configs.path_cropped_labels, subset, tile_label_name)
            cv2.imwrite(path_save_t, tile)
            cv2.imwrite(path_save_l, tile_label)
            count += 1

        # for the right bottom
        tile = image[size_original_h-size_cropped_h:size_original_h, size_original_w-size_cropped_w:size_original_w,:]
        tile_name = image_name + '_{}.tif'.format(count)

        tile_label = label[size_original_h - size_cropped_h:size_original_h, size_original_w - size_cropped_w:size_original_w,:]
        tile_label_name = label_name + '_{}.tif'.format(count)

        path_save_t = os.path.join(configs.path_cropped_images, subset, tile_name)
        path_save_l = os.path.join(configs.path_cropped_labels, subset, tile_label_name)

        cv2.imwrite(path_save_t, tile)
        cv2.imwrite(path_save_l, tile_label)
        count += 1


def tile_size_checker(configs, subset):

    path_images = glob.glob(os.path.join(configs.path_cropped_images, subset, '*'))
    path_labels = glob.glob(os.path.join(configs.path_cropped_labels, subset, '*'))

    for path_image, path_label in tqdm.tqdm(zip(path_images, path_labels)):

        image = cv2.imread(path_image)
        label = cv2.imread(path_label)

        assert tuple(image.shape) == tuple([256, 256, 3])
        assert image.shape == label.shape


def new_tile_cropper(configs, subset):
    """
    :param configs: config information of cropped images
    :param subset: subset name
    implement cropping with out the last row and column which need padding
    """
    assert subset == 'train' or subset == 'test', \
        'the name of subset should be in [train, test]'

    cr_w = configs.size_cropped_images_w
    cr_h = configs.size_cropped_images_h

    images = glob.glob(os.path.join(configs.path_source_images, subset, '*'))
    labels = glob.glob(os.path.join(configs.path_source_labels, subset, '*'))

    image_save_path = os.path.join(configs.path_cropped_images, subset)
    label_save_path = os.path.join(configs.path_cropped_labels, subset)

    ensure_dir(image_save_path)
    ensure_dir(label_save_path)

    h_overlap = int(cr_h / 2)
    w_overlap = int(cr_w / 2)
    for image_path, label_path in tqdm.tqdm(zip(images, labels)):
        image = Image.open(image_path)
        label = Image.open(label_path)
        count = 0
        image_name, image_ext = image_path.split('/')[-1].split('.')

        label_name, label_ext = label_path.split('/')[-1].split('.')
        image_w, image_h = image.size

        for start_h in range(0, image_h-cr_h, h_overlap):
            for start_w in range(0, image_w-cr_w, w_overlap):
                tile = image.crop((start_w, start_h, start_w+cr_w, start_h+cr_h))
                tile_label = label.crop((start_w, start_h, start_w+cr_w, start_h+cr_h))

                tile_name = image_name+'_{}.{}'.format(count, image_ext)
                tile_label_name = label_name+'_{}.{}'.format(count, label_ext)

                tile.save(os.path.join(configs.path_cropped_images, subset, tile_name))
                tile_label.save(os.path.join(configs.path_cropped_labels, subset, tile_label_name))

                count += 1












