# %%
import json
import numpy as np
import cv2

import os
import pickle
from typing import Tuple, List, Dict

# %%
CLASSES = {'un-classified': '4', 'destroyed': '3', 'major-damage': '2', 'minor-damage': '1', 'no-damage': '0'}

def check_directories(path: str) -> Tuple[str]:
    """Valida la integridad de los directorios.

    Args:
        path (str): ruta raíz a los directorios a validar.

    Returns:
        Tuple[str]: las tres rutas del conjunto de entrenamiento, prueba y validación.
    """
    if not os.path.isdir(path):
        raise Exception(f'there are not a valid path')

    train_path = os.path.join(path, 'train')
    if not os.path.isdir(train_path):
        raise Exception(f'You must add the train folder {train_path}')
    
    test_path= os.path.join(path, 'test')
    if not os.path.isdir(test_path):
        raise Exception(f'You must add the train folder {test_path}')
    
    val_path= os.path.join(path, 'hold')
    if not os.path.isdir(val_path):
        raise Exception(f'You must add the train folder {val_path}')
    
    return train_path, test_path, val_path


def json_to_yolo(_path: str, dst: str) -> Dict[str, List[np.ndarray]]:
    """Crea para cada imagen un archivo de etiqueta que el 
    modelo YOLO pueda entender:

    <clase> <coordenadas del poligono>

    Args:
        _path (str): ruta a las etiquetas originales.
        dst (str): ruta de destino a las etiquetas en formato YOLO.

    Returns:
        Dict[str, List[np.ndarray]]: contiene el par de 
        nombre del archivo contra la lista de los poligonos.
    """
    if not os.path.exists(dst):
        os.makedirs(dst)

    polygons_per_image = {}
    for file in os.listdir(_path):

        file_name, ext = os.path.splitext(file)
        if 'pre' in file_name:
            continue
        if ext == '.json':
            print(f'proceesing file {file_name}')
            with open(os.path.join(_path, file)) as f:
                j = json.load(f)

            polygons = []
            final = ""
            combined_file_name = file_name.replace('post', 'combined')
            new_txt = os.path.join(dst, f'{combined_file_name}.txt')
            
            with open(new_txt, 'w') as t:
                for _e in j['features']['xy']:
                    st_p = CLASSES[_e['properties']["subtype"]]
                    pairs = [coords for coords in _e['wkt'][10:-2].split(", ")]
                    nd_p = ""
                    coordinates = []
                    for pair in pairs:
                        x, y = pair.split(" ")
                        x = round(float(x), 4)
                        y = round(float(y), 4)
                        coordinates.append([x,y])
                        nd_p += " " + str(x/1024) + " " + str(y/1024)
                    final = st_p + nd_p
                    polygons.append(np.array(coordinates, dtype=int))
                    print(final)
                    t.write(final+'\n')
            print(f'file converted to YOLO format: {combined_file_name}.txt')
            polygons_per_image[combined_file_name] = polygons

    return polygons_per_image
            
def create_input_images(_path: str, dst: str):
    """Como entrada del modelo YOLO se crea una imagen superpuesta.
    Canal R: imagen antes del desastre.
    Canal G: imagen después del desastre.
    Canal B: Diferencia entre ambas imagenes.

    Args:
        _path (str): ruta a las imagenes originales.
        dst (str): ruta de destino a las imagenes en formato super puesto.
    """
    if not os.path.exists(dst):
        os.makedirs(dst)
    for pre_file in os.listdir(_path):
        if 'post' in pre_file:
            continue

        print(f'proccessing image: {pre_file}')
        post_file = pre_file.replace('pre', 'post')
        img = cv2.imread(os.path.join(_path, pre_file))
        img2 = cv2.imread(os.path.join(_path, post_file))
        g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        in_ = np.stack((g_img, g_img2, g_img2-g_img), axis=2)

        combined_image_name = pre_file.replace('pre', 'combined')
        combined_path = os.path.join(dst, combined_image_name)
        
        print(f'merged into: {combined_path}')
        cv2.imwrite(combined_path, in_)


# %%
def do_format(path: str):
    """Da el formato necesario al dataset para la entrada del modelo.

    Args:
        path (str): ruta raiz del dataset del modelo
    """
    paths = check_directories(path)
    
    steps = ('train', 'test', 'val')

    _data = os.path.join(path, 'data')
    _images = os.path.join(_data, 'images')
    _labels = os.path.join(_data, 'labels')

    for path_, step in zip(paths, steps):
        print(f'processing path: {path_} for step {step}')
        original_images = os.path.join(path_, 'images')
        create_input_images(original_images, os.path.join(_images, step))

        original_labels = os.path.join(path_, 'labels')
        label_dict = json_to_yolo(original_labels, os.path.join(_labels, step))
        with open(os.path.join(PATH, f'{step}.pickle'), 'wb') as f:
            pickle.dump(label_dict, f)


# %%
def cut_dataset(_path: str, percentage: float = 1/3):
    """Recorta el dataset a cierta proporción.

    Args:
        _path (str): ruta del dataset a recortar.
        percentage (float, optional): Defaults to 1/3.
    """
    for step in ('train', 'test', 'val'):
        step_path = os.path.join(_path, step)

        files = sorted(os.listdir(step_path))
        threshold = len(files)*percentage
        print(f'threshold for {step}: {threshold}')
        for c, file in enumerate(files):
            if c > threshold:
                file_to_remove = os.path.join(step_path, file)
                print(f'removing: {file_to_remove}')
                os.remove(file_to_remove) 
