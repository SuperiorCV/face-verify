import cv2,os,sys

DIR=os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(DIR,'../'))
import imageio
import Pytorch_Retinaface
import  MobileFaceNet_Pytorch
class Face_Modal():
    def __init__(self):
        self.face_detector = Pytorch_Retinaface.detect.FaceDector()
        self.faceVector_computer=MobileFaceNet_Pytorch.compute_face_vec.FaceVectorComputer()
    def register_face_check(self,id,img_path):
        flag,img,detect_result,points= self.face_detector.detect(img_path)
        if flag=='success':
            print(img.shape)
            face_aligin=self.face_detector.aligin(img,detect_result,points)
            face_vec=self.compute_face_vector(face_aligin)
            
            imageio.imwrite(f'./register_data/{id}.jpg',face_aligin)
            log=f'{id} 注册成功'
        elif flag=='error':
            log=f'图片不符合要求，有多个人脸'
        
           
        return flag,log
    def compute_face_vector(self,img):
        vec=self.faceVector_computer.compute_img(img)
        return vec