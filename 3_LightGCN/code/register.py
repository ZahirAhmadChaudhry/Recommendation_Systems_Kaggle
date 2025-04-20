"""
Created on Mar 1, 2020
Modified for Carrefour Dataset Implementation
"""
import world
import dataloader
import model
from pprint import pprint

if world.dataset in ['carrefour']:
    dataset = dataloader.CarrefourDataset()
else:
    raise NotImplementedError(f"Haven't supported {world.dataset} yet!")

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print('===========end===================')

# Register models and model names
MODELS = {
    'lgn': model.LightGCN
}

MODELS_NAME = {
    'lgn': 'LightGCN'
}
