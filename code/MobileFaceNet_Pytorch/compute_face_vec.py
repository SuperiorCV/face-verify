import sys,os
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
from core import model
import numpy as np
import torch
import imageio
import cv2
import torch.nn.functional as F
class FaceVectorComputer:
    def __init__(self,resume=os.path.join(__dir__,'./model/best/068.ckpt'),gpu=True):
        
        net = model.MobileFacenet()
        if gpu:
            net = net.cuda()
        if resume:
            print(type(resume))
            ckpt = torch.load(resume)
            net.load_state_dict(ckpt['net_state_dict'])
        net.eval()

        self.net=net
        self.gpu=gpu
    def compute(self,img):
        img=img.cuda()
        feature = None
    
        res = net(img).data.cpu().numpy()
        featureL = np.concatenate((res[0], res[1]), 1)
        featureR = np.concatenate((res[2], res[3]), 1)
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = np.concatenate((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = np.concatenate((featureRs, featureR), 0)
    def compute_img_by_path(self,img_path):
        img=imageio.imread(img_path)
        img= (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)
        img=torch.from_numpy(img).float().cuda()
        img=img.unsqueeze(0)
        vec=self.net(img)
        return vec
    def compute_img(self,img):
        img= (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)
        img=torch.from_numpy(img).float().cuda()
        img=img.unsqueeze(0)
        vec=self.net(img)
        return vec
if __name__=='__main__':
    fv_cmp=FaceVectorComputer()
    img1_path='/home/wht/face/code/MobileFaceNet_Pytorch/tmp/1.jpg'
    img2_path='/home/wht/face/code/MobileFaceNet_Pytorch/tmp/2.jpg'
    img3_path='/home/wht/face/code/MobileFaceNet_Pytorch/tmp/1_flip.jpg'
    vec1=fv_cmp.compute_img(img1_path)
    vec2=fv_cmp.compute_img(img2_path)
    vec3=fv_cmp.compute_img(img3_path)

    output1 = F.cosine_similarity(vec1,vec2)
    output2 = F.cosine_similarity(vec1,vec3)
    output3 = F.cosine_similarity(vec2,vec3)
    print(output1)
    print(output2)
    print(output3)

   
