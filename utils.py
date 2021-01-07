import os
import math
import random
import time
import torch
import cv2
import numpy as np

from contextlib import contextmanager
from sklearn.metrics import roc_auc_score


def get_score(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        score = roc_auc_score(y_true[:,i], y_pred[:,i])
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score, scores


@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def init_logger(log_file='./train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# seed_torch(seed=CFG.seed)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def reshape_img(cl_img):
    ht, wd = cl_img.shape
    ww = max(ht+2,wd+2)
    hh = ww
    constant = np.zeros((hh,ww), dtype=np.uint8)
    xx = (ww-wd)//2
    yy = (hh-ht)//2
    constant[yy:yy+ht, xx:xx+wd] = cl_img

    target_area = 1024*1024
    ratio = float(constant.shape[1])/float(constant.shape[0])
    new_h = int(np.sqrt(target_area / ratio) + 0.5)
    new_w = int((new_h * ratio) + 0.5)

    res_img = cv2.resize(constant, (new_w,new_h))
    return res_img


def NeedleAugmentation(image, n_needles=2, dark_needles=False, p=0.5, needle_folder='needle_aug'):
    aug_prob = random.random()
    if aug_prob < p:
        height, width, _ = image.shape  # target image width and height
        needle_images = [im for im in os.listdir(needle_folder) if 'png' in im]

        for _ in range(1, n_needles):
            needle = cv2.cvtColor(cv2.imread(os.path.join(needle_folder, random.choice(needle_images))), cv2.COLOR_BGR2RGB)
            needle = cv2.flip(needle, random.choice([-1, 0, 1]))
            needle = cv2.rotate(needle, random.choice([0, 1, 2]))

            h_height, h_width, _ = needle.shape  # needle image width and height
            roi_ho = random.randint(0, abs(image.shape[0] - needle.shape[0]))
            roi_wo = random.randint(0, abs(image.shape[1] - needle.shape[1]))
            roi = image[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

            # Creating a mask and inverse mask 
            img2gray = cv2.cvtColor(needle, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # Now black-out the area of needle in ROI
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # Take only region of insect from insect image.
            if dark_needles:
                img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                needle_fg = cv2.bitwise_and(img_bg, img_bg, mask=mask)
            else:
                needle_fg = cv2.bitwise_and(needle, needle, mask=mask)

            # Put needle in ROI and modify the target image
            dst = cv2.add(img_bg, needle_fg, dtype=cv2.CV_64F)

            image[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst

    return image
