import random
import numpy as np 
from utils.grasp import asGraspRectangle, asGrasp
from PIL import Image, ImageDraw, ImageEnhance

def _mask2bbox(mask):
    mask_np = np.array(mask)
    H, W = mask_np.shape
    x_min, x_max, y_min, y_max = W-1, 0, H-1, 0
    
    for i in range(W):
        if 1 in mask_np[:, i]:
            if x_min > i:
                # print(i)
                x_min = i
            if x_max < i:
                x_max = i
    for i in range(H):
        if 1 in mask_np[i, :]:
            if y_min > i:
                y_min = i
            if y_max < i:
                y_max = i
    # print("BBox:", x_min, y_min, x_max, y_max)
    return np.array([x_min, y_min, x_max, y_max])

def _rotGrasp(refgrasp, rdm_a, img_height, img_width):
    H, W = img_height, img_width
    c_x_o, c_y_o = W /2, H/2  # 图片中心
    # print(rdm_a)
    b = np.deg2rad(rdm_a) 
    rects = [asGraspRectangle(grasp_box) for grasp_box in refgrasp]
    rect_news = []

    for r in rects: 
        rect_new = []
        for p in r:
            c_x_new = (p[0] - c_x_o) * np.cos(b) - (p[1] - c_y_o) * np.sin(b) + c_x_o
            c_y_new = (p[0] - c_x_o) * np.sin(b) + (p[1] - c_y_o) * np.cos(b) + c_y_o

            rect_new.append([round(c_x_new), round(c_y_new)])

        rect_news.append(rect_new)

    # print(rect_news)
    if (np.array(rect_news) < 0).mean() > 0 \
        or (np.array(rect_news)[:,:,0] >= img_width).mean() > 0 \
        or (np.array(rect_news)[:,:,1] >= img_height).mean() > 0:

        return False

    refgrasp_new = asGrasp(np.array(rect_news))

    return np.array(refgrasp_new)

def crop(rgb_img, depth_img, refmask, refbbox, refgrasp, probability=0.5):
    rand = random.random()
    if rand > probability:
        return rgb_img, depth_img, refmask, refbbox, refgrasp 
    
    if rgb_img is not None:
        W, H = rgb_img.size
    else:
        W, H = depth_img.size

    if refbbox is not None:
        # print(refbbox)
        min_x, min_y, max_x, max_y = refbbox
    else:
        rect = []
        for gr in refgrasp:
            rect.append(asGraspRectangle(gr))
        rect = np.array(rect)
        rect_T = np.swapaxes(rect, 1, 2)
        min_x, min_y, max_x, max_y = rect_T[:,0,:].min(), rect_T[:,1,:].min(), rect_T[:,0,:].max(), rect_T[:,1,:].max()
    
    dx1, dy1, dx2, dy2 = random.sample(range(0, min(W, H)//4), k=4)

    x1_crop = dx1 if dx1 < min_x else min_x-1
    y1_crop = dy1 if dy1 < min_y else min_y-1
    x2_crop = W - 1 - dx2 if W - 1 - dx2 > max_x else max_x+1
    y2_crop = H - 1 - dy2 if H - 1 - dy2 > max_y else max_y+1
    # print((x1_crop, y1_crop, x2_crop, y2_crop))
    if rgb_img is not None:
        rgb_img = rgb_img.crop((x1_crop, y1_crop, x2_crop, y2_crop))
    if depth_img is not None:
        depth_img = depth_img.crop((x1_crop, y1_crop, x2_crop, y2_crop))
    if refmask is not None:
        refmask = refmask.crop((x1_crop, y1_crop, x2_crop, y2_crop))
    if refbbox is not None:
        refbbox = np.array(refbbox) - np.array([x1_crop, y1_crop, x1_crop, y1_crop])

    refgrasp = np.array(refgrasp) - np.array([x1_crop, y1_crop, 0 ,x1_crop, y1_crop])

    return rgb_img, depth_img, refmask, refbbox, refgrasp 

def bright_contrast_color(rgb_img, probability=0.5):
    if rgb_img is None:
        return None
    img = rgb_img

    if random.random() > probability:
        return img

    bright = random.uniform(0.6, 1.4)
    contrast = random.uniform(0.9, 1.5)
    color = random.uniform(0.8, 1.5)

    random_seed = random.random()
    if random_seed > 0.5:
        brightEnhancer = ImageEnhance.Brightness(img)
        img = brightEnhancer.enhance(bright)

    random_seed = random.random()
    if random_seed > 0.5:    
        contrastEnhancer = ImageEnhance.Contrast(img)
        img = contrastEnhancer.enhance(contrast)

    random_seed = random.random()
    if random_seed > 0.5:
        colorEnhancer = ImageEnhance.Color(img)
        img = colorEnhancer.enhance(color)
    return img

def random_rotate(rgb_img, depth_img, refmask, refbbox, refgrasp, probability=0.5):

    if random.random() > probability:
        return rgb_img, depth_img, refmask, refbbox, refgrasp

    rgb_img_old, depth_img_old, refmask_old, refbbox_old, refgrasp_old = rgb_img, depth_img, refmask, refbbox, refgrasp
    max_angle = 15
    
    rdm_a = random.uniform(-max_angle, max_angle)
    if rgb_img is not None:
        W, H = rgb_img.size
        rgb_img = rgb_img.rotate(-rdm_a)
        
    if depth_img is not None:
        W, H = depth_img.size
        depth_img = depth_img.rotate(-rdm_a)

    if refmask is not None:
        refmask = refmask.rotate(-rdm_a)


    refgrasp = _rotGrasp(refgrasp, rdm_a, H, W)

    if refbbox is not None:
        refbbox = _mask2bbox(refmask)

    if refgrasp is False:
        return rgb_img_old, depth_img_old, refmask_old, refbbox_old, refgrasp_old

    return rgb_img, depth_img, refmask, refbbox, refgrasp


def coarse_erasing(rgb_img, min_h_ratio=0.001, max_h_ratio=0.02, w_aspect_ratio=0.6, eras_num=20, probability = 0.2):
    '''
        this function can erase some area and fill in different color
        probability: The probability that the operation will be performed.
        sl: min erasing area
        sh: max erasing area
        r1: min aspect ratio
        mean: erasing value
    '''
    if rgb_img is None:
        return None
    

    random_seed = random.random()
    if random_seed > probability:
        return rgb_img
    
    img = np.array(rgb_img)
    H, W, C = img.shape
    at = random.choice(range(0, eras_num))
    for _ in range(at):

        a1 = random.uniform(0, 1) 
        a2 = random.uniform(0, 1) 
        a3 = random.uniform(0, 1) 
        mean=[round(a1*255), round(a2*255), round(a3*255)]

        target_H_ratio = random.uniform(min_h_ratio, max_h_ratio) 
        aspect_ratio = random.uniform(w_aspect_ratio, 1/w_aspect_ratio)

        h = int(round(target_H_ratio * H))
        w = int(round(h * aspect_ratio))

        if w < W and h < H:
            x1 = random.randint(0, H - h)
            y1 = random.randint(0, W - w)
            if C == 3:
                img[x1:x1+h, y1:y1+w, 0] = mean[0]
                img[x1:x1+h, y1:y1+w, 1] = mean[1]
                img[x1:x1+h, y1:y1+w, 2] = mean[2]
            else:
                img[x1:x1+h, y1:y1+w, 0] = mean[0]
    img = Image.fromarray(img)
    
    return img

def FLIP(rgb_img, depth_img, sentence, refmask, refbbox, refgrasp, probability=0.2):
    W, H = rgb_img.size
    random_seed = random.random()
    if random_seed > probability:
        return rgb_img, depth_img, sentence, refmask, refbbox, refgrasp
    if rgb_img is not None:
        rgb_img = np.array(rgb_img)
        rgb_img = np.fliplr(rgb_img)
        rgb_img = Image.fromarray(rgb_img)
    
    if depth_img is not None:
        depth_img = np.fliplr(depth_img)
    
    if refmask is not None:
        refmask = np.array(refmask)
        refmask = np.fliplr(refmask)
        refmask = Image.fromarray(refmask)
    if sentence is not None:
        if 'left' in sentence:
            sentence = sentence.replace('left', 'right')
        elif 'right' in sentence:
            sentence = sentence.replace('right', 'left')
    if refbbox is not None:
        old_xmin, old_ymin, old_xmax, old_ymax = refbbox
        new_xmin = W - old_xmax
        new_xmax = W - old_xmin
        new_ymin = old_ymin
        new_xmax = old_ymax

        refbbox = np.array([new_xmin, new_ymin, new_xmax, new_xmax])
    
    if refgrasp is not None:
        new_refgrasp = []
        for g in refgrasp:
            old_x_min, old_y_min, old_theta, old_x_max, old_y_max = g
            new_x_min = W - old_x_max
            new_x_max = W - old_x_min
            new_y_min = old_y_min
            new_y_max = old_y_max
            new_theta = 180.0 - old_theta
            new_refgrasp.append([new_x_min, new_y_min, new_theta, new_x_max, new_y_max])
        refgrasp = np.array(new_refgrasp)

    return rgb_img, depth_img, sentence, refmask, refbbox, refgrasp

def ereasingWord(sentence, entity_list, adj_list, max_era=2, probability=0.2):
    random_seed = random.random()
    if random_seed > probability:
        return sentence
    sentence = sentence.strip('.')

    
    words = sentence.split(" ")
    
    words = [word.replace("_", " ") for word in words]
    era_num = 0
    
    for i in range(max_era+3):
        idx = random.choice(range(len(words)))
        if words[idx] in entity_list or  words[idx] in adj_list:
            continue

        words.pop(idx)

        sentence = " ".join(words) + '.'

        era_num += 1
        if era_num >= random.choice(range(1, max_era+1)):
            break
    return sentence


def replaceWord(sentence, entity_list, probability=0.5):
    random_seed = random.random()
    if random_seed > probability:
        return sentence
    sentence = sentence.strip('.')

    
    words = sentence.split(" ")
    
    words = [word.replace("_", " ") for word in words]

    for i, w in enumerate(words):
        if w in entity_list:
            words[i] = 'object'
            break

    sentence = " ".join(words) + '.'
    return sentence