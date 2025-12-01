# %%
from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
from typing import List, Tuple

# Load a pretrained YOLO11n model
#first
#model_path = 'runs/segment 2/train/weights/last.pt'
#last
model_path = 'runs/segment 2/train/weights/last.pt'


# %%
CLASSES = {'un-classified': '4', 'destroyed': '3', 'major-damage': '2', 'minor-damage': '1', 'no-damage': '0'}
CLASS_COLORS = {
    'un-classified': (93, 41, 84), 
    'destroyed': (95, 51, 131), 
    'major-damage': (111, 72, 188), 
    'minor-damage': (255, 107, 128), 
    'no-damage': (255, 163, 144)}

# %%
model = YOLO(model_path)

# %%
STEP = 'test'
# combined post processed images
COMBINED_IMG_PATH = f'resources/data/images/{STEP}/'
COMBINED_LABEL_PATH = f'resources/data/labels/{STEP}/'
# original path to images and labels
ORIGINAL_IMG_PATH = f'resources/{STEP}/images/'
ORIGINAL_LABEL_PATH = f'resources/{STEP}/labels/'
# list of combined images
combined_images = sorted(os.listdir(COMBINED_IMG_PATH))


def plot_bboxes(img: np.ndarray, results: list, mask: bool = 'colors') -> Tuple[np.ndarray]:
    """Generate annotation and mask images.

    Args:
        img (np.ndarray): 1024x1024 img to draw in.
        results (list): list of results of predictions of model.
        mask (bool, optional): if white mask must be draw in colors. Defaults to 'colors'.

    Returns:
        Tuple[np.ndarray]: Annotation and mask images.
    """
    black_background_img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    names = results[0].names # class names dict
    scores = results[0].boxes.conf.numpy() # probabilities
    classes = results[0].boxes.cls.numpy() # predicted classes
    boxes = results[0].boxes.xyxy.numpy().astype(np.int32) # bboxes
    for score, cls, bbox in zip(scores, classes, boxes): # loop over all bboxes

        class_label = names[cls] # class name
        color = CLASS_COLORS[class_label]
        label = f"{class_label} : {score:0.2f}" # bbox label
        lbl_margin = 3 #label margin
        img = cv2.rectangle(img, (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            color=color,
                            thickness=2)
        label_size = cv2.getTextSize(label, # labelsize in pixels 
                                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                     fontScale=1, thickness=1)
        lbl_w, lbl_h = label_size[0] # label w and h
        lbl_w += 2* lbl_margin # add margins on both sides
        lbl_h += 2*lbl_margin
        img = cv2.rectangle(img, (bbox[0], bbox[1]), # plot label background
                             (bbox[0]+lbl_w, bbox[1]-lbl_h),
                             color=color, 
                             thickness=-1) # thickness=-1 means filled rectangle
        cv2.putText(img, label, (bbox[0]+ lbl_margin, bbox[1]-lbl_margin), # write label to the image
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0, color=(0, 0, 0),
                    thickness=3)
        
        # generate mask:
        black_background_img = cv2.rectangle(black_background_img, (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            color=color if mask == 'colors' else (255, 255, 255), 
                            thickness=-1)
    return img, black_background_img


# %%
def generate_prediction(combined_img_number: int):
    """Plot annotation image over image.

    Args:
        combined_img_number (int): number of combined image to generate predictions.
    """
    combined_img_name = combined_images[combined_img_number]
    combined_path = COMBINED_IMG_PATH + combined_img_name
    post_img_name = combined_img_name.replace('combined', 'post')

    combined_img = cv2.imread(combined_path)
    post_image = cv2.imread(ORIGINAL_IMG_PATH+post_img_name)

    results = model(combined_img)

    img, mask = plot_bboxes(post_image, results, mask='colors')
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# %%
def generate_predictions_and_mask(s_folder: str = 'processed_data/test_2/'):
    """Generates annotation and mask images for test dataset images.

    Args:
        s_folder (str, optional): folder to save prediction results. Defaults to 'processed_data/test/'.
    """
    combined_images = sorted(os.listdir(COMBINED_IMG_PATH))
    for combined_img_name in combined_images:
        image_path = COMBINED_IMG_PATH + combined_img_name

        post_img_name = combined_img_name.replace('combined', 'post')

        original_image = cv2.imread(ORIGINAL_IMG_PATH+post_img_name)

        img = cv2.imread(image_path)
        results = model(img)
        
        img, mask = plot_bboxes(original_image, results, mask='white')
        s_folder_mask = s_folder+'masks/'
        s_folder_seg = s_folder+'segmentation/'
        if not os.path.exists(s_folder_mask):
            os.makedirs(s_folder_mask)
        if not os.path.exists(s_folder_seg):
            os.makedirs(s_folder_seg)
        cv2.imwrite(s_folder_mask+post_img_name, mask)
        cv2.imwrite(s_folder_seg+post_img_name, img)


# %%
def generate_rects(label_path: str, save: bool= False, file_path_ts:str='processed_data/test/', _generate_rects:bool = True) -> Tuple[List[List[int]], List[int]]:
    """Generate rectangle coordinates for each poligon in label.

    Args:
        label_path (str): path to label of test dataset.
        save (bool, optional): if transformed rects must be saved. Defaults to False.
        file_path_ts (str, optional): file path to save. Defaults to 'processed_data/test/'.
        _generate_rects (bool, optional): if rects must be generated. Defaults to True.

    Returns:
        Tuple[List[List[int]], List[int]]: _description_
    """
    if not _generate_rects:
        with open(f'{file_path_ts}predicted_rects.pkl', 'rb') as p_rcts:
            rects = pickle.load(p_rcts) # deserialize using load()

        with open(f'{file_path_ts}predicted_cls.pkl', 'rb') as p_cls:
            original_cls = pickle.load(p_cls)

    else:
        with open(label_path) as f:
            file = f.read()

        print(label_path)
        
        original_cls = [int(l[0]) for l in file.split('\n') if l]
        raw_nums = [[round(float(n), 4)*1024 for n in l[2:].split()] for l in file.split('\n') if l]
        txt = [[p for p in zip(row[::2], row[1::2])] for row in raw_nums]

        rects = []
        for line in txt:
            x0 = float('inf')
            y0 = float('inf')
            x1 = -1
            y1 = -1
            for pair in line:
                x = pair[0]
                y = pair[1]
                if x < x0:
                    x0 = x
                if y < y1:
                    y0 = y
                if x > x1:
                    x1 = x
                if y > y1:
                    y1 = y
            rects.append([x0, y0, x1, y1])

        if save:
            with open(file_path_ts+'predicted_rects.pkl', 'wb') as f:
                pickle.dump(rects, f)
            with open(file_path_ts+'predicted_cls.pkl', 'wb') as f:
                pickle.dump(original_cls, f)
        
        
    return rects, original_cls

def pair_boxes(results_: list, rects: List[List[int]], original_cls: List[int], threshold: int = 55) -> List[int]:
    """Generate a list of response classes paired with original ones.

    Args:
        results_ (list): lists of results of YOLO prediction.
        rects (List[List[int]]): list of rectangle coordinates.
        original_cls (List[int]): list of original classes.
        threshold (int, optional): threshold to pair original with response rectangle. Defaults to 55.

    Returns:
        List[int]: list of response classes.
    """
    response_cls = [-1 for _ in original_cls]
    classes = results_[0].boxes.cls.numpy().tolist() # predicted classes
    boxes = results_[0].boxes.xyxy.numpy().astype(np.int32).tolist() # bboxes
    for i, o_rect in enumerate(rects):
        min_distance = float('inf')
        _del = -1
        for j, (bbox, cls) in enumerate(zip(boxes, classes)): # loop over all bboxes
            distance = abs(o_rect[0] - bbox[0])
            distance += abs(o_rect[1] - bbox[1])
            distance += abs(o_rect[2] - bbox[2])
            distance += abs(o_rect[3] - bbox[3])

            if distance < min_distance and distance < threshold:
                # print(distance)
                response_cls[i] = int(cls)
                _del = j
        if len(boxes) > 0 and _del != -1:
            boxes.pop(_del)
            classes.pop(_del)

    return response_cls
    

def generate_confusion_matrix(_generate_rects:bool=True, path_to_sd ='processed_data/test/'):
    """Generates confusion matrix.

    Args:
        _generate_rects (bool, optional): if generate rectangles or load from saved file. Defaults to True.
        path_to_sd (str, optional): path to saved_. Defaults to 'processed_data/test/'.
    """
    final_response_cls = []
    final_original_cls = []
    for img_name in os.listdir(COMBINED_IMG_PATH):
        img_name = img_name.replace('.png', '')
        img_ = cv2.imread(f'{COMBINED_IMG_PATH}{img_name}.png')
        label_path = f'{COMBINED_LABEL_PATH}{img_name}.txt'

        results_ = model(img_)
        
        rects, original_cls = generate_rects(label_path, save=True, file_path_ts=path_to_sd, _generate_rects = _generate_rects)

        response_cls = pair_boxes(results_, rects, original_cls)

        final_original_cls += original_cls
        final_response_cls += response_cls

    cm = confusion_matrix(final_original_cls, final_response_cls)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=[-1,0,1,2,3,4],)
    {'un-classified': '4', 'destroyed': '3', 'major-damage': '2', 'minor-damage': '1', 'no-damage': '0'}
    disp.plot(cmap="Blues")

