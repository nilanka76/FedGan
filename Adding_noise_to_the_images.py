
import cv2
import os
from progressbar import ProgressBar
from skimage.util import random_noise

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, 0o755)

bar = ProgressBar()

origin_dir = 'PATH for Original Medical_images'
goal_dir = 'PATH for Original Medical_images_processed'
dataset_build = True
size =256

if not os.path.exists(origin_dir) and not dataset_build:
    print('please check your input dir')
    exit()

if not dataset_build:
    dir_list = os.listdir(origin_dir)

if not os.path.exists(goal_dir):
    make_dir(goal_dir)

if dataset_build:
    print('done')
    print('start building dataset')
    make_dir(goal_dir + '/1_train/clean')
    make_dir(goal_dir + '/1_train/noise15')
    make_dir(goal_dir + '/1_train/noise25')
    make_dir(goal_dir + '/1_train/noise50')

    make_dir(goal_dir + '/2_val/clean')
    make_dir(goal_dir + '/2_val/noise15')
    make_dir(goal_dir + '/2_val/noise25')
    make_dir(goal_dir + '/2_val/noise50')

    make_dir(goal_dir + '/3_test/clean')
    make_dir(goal_dir + '/3_test/noise15')
    make_dir(goal_dir + '/3_test/noise25')
    make_dir(goal_dir + '/3_test/noise50')


    dir_list = os.listdir(target_path)
    # resize image and add noise
    for i in bar(range(len(dir_list[:500]))):
        img = cv2.imread(os.path.join(target_path, dir_list[i]))
        x, y = img.shape[0:2]
        if not all([x == size, y == size]):
            img = cv2.resize(img, dsize=(size, size))
        if i >= 400:
            tmp_dir = '/3_test'
        elif i < 400 and i >= 350:
            tmp_dir = '/2_val'
        else:
            tmp_dir = '/1_train'
        cv2.imwrite(os.path.join(goal_dir + tmp_dir + '/clean/', dir_list[i]), img)
        for n in [15, 25, 50]:
            noise = random_noise(img / 255.0, mode='gaussian', var=(n / 255.0) ** 2)
            cv2.imwrite(os.path.join(goal_dir + tmp_dir + '/noise' + str(n) + '/', dir_list[i]),
                        (noise * 255.0).astype(int))
    print('noise dataset build finish ,you can find your dataset in < ' + goal_dir + ' >')
    exit()

