"""
@author: JiXuan Xu, Jun Wang
@date: 20201023
@contact: jun21wangustc@gmail.com 
"""

import torch

from core.model_loader.BaseModelLoader import BaseModelLoader

class FaceAlignModelLoader(BaseModelLoader):
    def __init__(self, model_path, model_category, model_name, meta_file='model_meta.json'):
        print('Start to analyze the face landmark model, model path: %s, model category: %s，model name: %s' %
                    (model_path, model_category, model_name))
        super().__init__(model_path, model_category, model_name, meta_file)
        self.cfg['img_size'] = self.meta_conf['input_width']
        
    def load_model(self):
        try:
            model = torch.load(self.cfg['model_file_path'])
        except Exception as e:
            print('The model failed to load, please check the model path: %s!'
                         % self.cfg['model_file_path'])
            raise e
        else:
            print('Successfully loaded the face landmark model!')
            return model, self.cfg
