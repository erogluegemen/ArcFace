import os
import shutil
import torch
import numpy as np
from data.eval_lfw import evaluation_10_fold, getFeatureFromTorch

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,'model_best.pth.tar'))
        print('best model saved\n')


def test(conf,net,lfwdataset,lfwloader):
	getFeatureFromTorch(conf.save_dir+'/cur_lfw_result.mat', net, conf.device, lfwdataset, lfwloader)
	lfw_acc = np.mean(evaluation_10_fold(conf.save_dir+'/cur_lfw_result.mat'))

	return lfw_acc