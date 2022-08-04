#%%
from genericpath import exists
import os
import random
from unicodedata import category
# import cv2
import numpy as np
# from numpy.lib.function_base import append
# import torch
# import torchvision
import json
# import requests
import datetime
import yaml

print(f'start merge tool version 1.0')
#%%
json_files = [
    '../../../datasets/cauda_test/dic_1004/anno.json',
    '../../../datasets/cauda_test/dic_1008/anno.json',
    '../../../datasets/cauda_test/dic_1009/anno.json'
]
save_name = './temp/merge.json'

#%%
import argparse
parser = argparse.ArgumentParser(description="coco mergy")
parser.add_argument('--config','-c', type=str, 
    help='help : config file path, ex> cmd.yaml')

parsed_args = parser.parse_args()

with open(parsed_args.config, 'r') as f:
    cmdConfig = yaml.load(f,Loader=yaml.FullLoader)

json_files = cmdConfig['merge']['list']
save_name = cmdConfig['merge']['save_name']


# %%
annotation = []
images = []
meta_datas = []
categories = []

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)
        _categories = data['categories']

        if(len(categories) == 0):
            images.extend(data['images'])
            meta_datas.extend(data['meta'])
            for cat in _categories: 
                if cat['supercategory'] != '':
                    cat['name'] = cat['supercategory'] + '_' + cat['name']
                categories.append(cat)
            annotation.extend(data['annotations'])
            
        else:
            last_cat_id = categories[-1]['id']
            
            # can_last = [meta_datas[-1]['id'],images[-1]['id'],annotation[-1]['id']]
            # print(can_last)
            # _last_id = max(can_last)
            # print(f'last_id: {last_id}')
            
            _cat_conv_table = []
            
            for cat in _categories: 
                
                if cat['supercategory'] != '':
                    _super_category_name = cat['supercategory'] + '_' + cat['name']
                else:
                    _super_category_name = cat['name']
                
                if _super_category_name in [c['name'] for c in categories]:
                    for _c in categories:
                        if _c['name'] == _super_category_name:
                            _cat_conv_table.append(cat['is'], _c['id'])
                            break
                    # _cat_conv_tabx/le.append([old_id,cat['id']])
                else :
                    old_id = cat['id']
                    cat['id'] += last_cat_id
                    cat['name'] = _super_category_name
                    categories.append(cat)
                    _cat_conv_table.append([old_id,cat['id']])
            
            meta_datas.extend(data['meta'])
            # for meta in data['meta']:
            #     meta_datas.append(meta)
            
            images.extend(data['images'])
            # for img in data['images']:
            #     images.append(img)
            
            for anno in data['annotations']:
                # anno['id'] += _last_id
                for i,j in _cat_conv_table:
                    if anno['category_id'] == i:
                        anno['category_id'] = j
                        break;
                # anno['category_id'] = _cat_conv_table[anno['category_id']-1][1]
                # anno['image_id'] += _last_id
                annotation.append(anno)
                
                # get last index


for _cat in categories:
    print(_cat)
        

#%%
dataset_dicts = {
        "info": {
            "description": "daisy ai solution",
            "url": "",
            "version": "1",
            "date_created": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        },
        "images": images,
        "annotations": annotation,
        "meta": meta_datas,
        "categories": categories,
    }

# %%
# save_name = '../temp/_data.json'
os.makedirs(os.path.dirname(save_name), exist_ok=True) # make dir if not exist
with open(save_name, 'w') as f:
    json.dump( 
        obj=dataset_dicts, 
        fp=f,
        indent=2 # 줄맞추기
)

print(f'{save_name} 저장 완료 , image num {len(dataset_dicts["images"])} , anno num {len(dataset_dicts["annotations"])}')

# %%
