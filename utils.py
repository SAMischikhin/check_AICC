import csv
import re
import pandas as pd
import numpy as np

def get_input_file_dict(input_dir_list, required_input_file_names): # list, list -> dict or None
    """возвращает имена требуемых инпут-файлов, или None, если нет хотя бы одного,
    возвращвется словарь: ключ: имя; значение: имя с расширением из input dir """
    input_dir_list_name = [a.split('.')[0] for a in input_dir_list]
    if set(required_input_file_names) <= set(input_dir_list_name):
        return {a.split('.')[0]: a for a in input_dir_list if a.split('.')[0] in required_input_file_names}

def get_volumes_from_BXDATA(path): #--> list, all volumes by boxes
    boxes_number_mask = re.compile(r'\s*Number_box\s=\s(\d*)\s*')
    boxes_volumes_mask = re.compile(r'\s*Volume\s=\s([0-9\s\t]*)\s*')
    volumes_string_mask = re.compile(r'[Ee+0-9.\s\t]*')
    volume_block_is = False
    volumes_arr = []
    with open(path) as f:
        for line in f:
            boxes_number = boxes_number_mask.search(line)
            if boxes_number:
                Nboxes = int(boxes_number.group(1))
                break  
        for line in f:
            boxes_volume = boxes_volumes_mask.search(line)
            if boxes_volume:
                volume_block_is = True
            if volume_block_is:
                volumes_string = volumes_string_mask.fullmatch(line)
                if volumes_string:
                    volumes_arr += re.compile('[Ee+0-9.]+').findall(volumes_string.group(0))
                    if len(volumes_arr) >= Nboxes:
                        break
    return volumes_arr

def find_delimiter(path):
    """ получение разделителя по первой строке файла """
    sniffer = csv.Sniffer()
    with open(path) as fp:
        delimiter = sniffer.sniff(fp.readline()).delimiter
    fp.close()
    return delimiter

def read_csv_decorator(func):
    """ декоратор для чтения файла csv с разделителем по первой строчки файла """
    def wrapper(*args):
        delimiter = find_delimiter(args[0]) 
        return func(*args, sep=delimiter, skipinitialspace=True)
    return wrapper
