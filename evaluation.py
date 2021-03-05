from test_util import test_net
import time
import file_utils
import os
import imgproc
import cv2
from eval.script import eval_2015
# from eval.script import eval_2013


def eval2013(craft, test_folder, result_folder, text_threshold=0.7, link_threshold=0.4, low_text=0.4):
    image_list, _, _ = file_utils.get_files(test_folder)
    t = time.time()
    res_gt_folder = os.path.join(result_folder, 'gt')
    res_mask_folder = os.path.join(result_folder, 'mask')
    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\n')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(craft, image, text_threshold, link_threshold, low_text, True, False, 980,
                                             1.5, False)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = os.path.join(res_mask_folder, "/res_" + filename + '_mask.jpg')
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult13(image_path, polys, dirname=res_gt_folder)

    eval_2013(res_gt_folder)
    print("elapsed time : {}s".format(time.time() - t))

def list_img_ic2015(test_img_file):
    test_img_dir = []
    for filename in os.listdir(test_img_file):
        test_img_dir.append(os.path.join(test_img_file, filename))
    return test_img_dir

def eval2015(craft, test_image_folder, result_folder, text_threshold=0.7, link_threshold=0.4, low_text=0.4):
    # image_list, _, _ = file_utils.get_files(test_folder)
    image_list = list_img_ic2015(test_image_folder)
    t = time.time()
    res_gt_folder = os.path.join(result_folder, 'gt')
    res_mask_folder = os.path.join(result_folder, 'mask')
    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\n')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(craft, image, text_threshold, link_threshold, low_text, True, True)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = os.path.join(res_mask_folder, "/res_" + filename + '_mask.jpg')
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, bboxes, dirname=res_gt_folder)

    eval_2015(res_gt_folder)
    print("elapsed time : {}s".format(time.time() - t))
