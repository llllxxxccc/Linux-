#!/usr/bin/env python
#encoding:utf-8
"""
@author:
@time:2017/3/18 21:03
Set the config variable.
"""
import configparser as cp
import json

config = cp.RawConfigParser()
config.read('./data/config/config.cfg')

orientations = json.loads(config.get("hog", "orientations"))
pixels_per_cell = json.loads(config.get("hog", "pixels_per_cell"))
cells_per_block = json.loads(config.get("hog", "cells_per_block"))
visualize = config.getboolean("hog", "visualize")
normalize = config.getboolean("hog", "normalize")

train_feat_path = config.get("path", "train_feat_path")
test_feat_path = config.get("path", "test_feat_path")
model_path = config.get("path", "model_path")
model_path_2 = config.get("path", "model_path_2")
model_path_3 = config.get("path", "model_path_3")
model_path_4 = config.get("path", "model_path_4")
model_path_5 = config.get("path", "model_path_5")

train_feat_path1 = config.get("path", "train_feat_path1")
test_feat_path1 = config.get("path", "test_feat_path1")
model_path1 = config.get("path", "model_path1")

train_feat_path2 = config.get("path", "train_feat_path2")
test_feat_path2 = config.get("path", "test_feat_path2")
model_path2= config.get("path", "model_path2")
model_path2_2= config.get("path", "model_path2_2")
model_path2_3= config.get("path", "model_path2_3")
model_path2_4= config.get("path", "model_path2_4")
model_path2_5= config.get("path", "model_path2_5")

train_feat_path3 = config.get("path", "train_feat_path3")
test_feat_path3 = config.get("path", "test_feat_path3")
model_path3= config.get("path", "model_path3")

train_feat_path4 = config.get("path", "train_feat_path4")
test_feat_path4 = config.get("path", "test_feat_path4")
model_path4= config.get("path", "model_path4")

train_feat_path5 = config.get("path", "train_feat_path5")
test_feat_path5 = config.get("path", "test_feat_path5")
model_path5= config.get("path", "model_path5")

train_feat_path6 = config.get("path", "train_feat_path6")
test_feat_path6 = config.get("path", "test_feat_path6")
model_path6= config.get("path", "model_path6")