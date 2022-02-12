from __future__ import print_function
import os,sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
# sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
from data import cfg_mnet, cfg_re50

from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
import yaml


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class FaceDector():
    def __init__(self,config_path=os.path.join(__dir__,'./config.yaml')):
        import yaml
        yaml_file = open(config_path)  #  传入文件路径
        self.args = yaml.load(yaml_file,Loader=yaml.FullLoader)
        print(self.args)
        torch.set_grad_enabled(False)
        cfg = None
        if self.args['FaceDetector']['network'] == "mobile0.25":
            cfg = cfg_mnet
        elif self.args['FaceDetector']['network'] == "resnet50":
            cfg = cfg_re50
        net = RetinaFace(cfg=cfg, phase = 'test')
        net = load_model(net, self.args['FaceDetector']['trained_model'], self.args['FaceDetector']['cpu'])
        net.eval()
        print('Finished loading model!')
        print(net)
        cudnn.benchmark = True
        device = torch.device("cpu" if self.args['FaceDetector']['cpu'] else "cuda")
        net = net.to(device)

        self.resize=1
        self.cfg=cfg
        self.net=net
        self.device=device
    
        # args = dict(
        #     trained_model='./weights/mobilenet0.25_Final.pth',
        #     network='mobile0.25',
        #     cpu=False,
        #     confidence_threshold=0.02,
        #     top_k=5000,
        #     nms_threshold=0.4,
        #     keep_top_k=750,
        #     save_image=True,
        #     vis_thres=0.6
        # )
        # parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
        #                     type=str, help='Trained state_dict file path to open')
        # parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
        # parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
        # parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
        # parser.add_argument('--top_k', default=5000, type=int, help='top_k')
        # parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
        # parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
        # parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
        # parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
        # args = parser.parse_args()


    def detect(self,image_path="./curve/tmp.jpg"):
        print('image_path',image_path)
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        h,w,_=img_raw.shape
        tmp_scale=1.0
        if h>2000 or w>2000:
            tmp_scale=0.25
        elif h>1000 or w>1000:
            tmp_scale=0.5
        img_raw = cv2.resize(img_raw, (0, 0), fx=tmp_scale, fy=tmp_scale, interpolation=cv2.INTER_NEAREST)
        img_raw_source=img_raw.copy()
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        tic = time.time()
        loc, conf, landms = self.net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.args['FaceDetector']['confidence_threshold'])[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.args['FaceDetector']['top_k']]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.args['FaceDetector']['nms_threshold'])
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.args['FaceDetector']['keep_top_k'], :]
        landms = landms[:self.args['FaceDetector']['keep_top_k'], :]

        dets = np.concatenate((dets, landms), axis=1)

        # show image
        count_face=0
        if self.args['FaceDetector']['save_image']:
            for b in dets:
                if b[4] < self.args['FaceDetector']['vis_thres']:
                    continue
                else:
                    count_face+=1
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image

            name = "test.jpg"
            cv2.imwrite(name, img_raw)


         #到这里为止我们已经利用Retinaface_pytorch的预训练模型检测完了人脸，并获得了人脸框和人脸五个特征点的坐标信息，全保存在dets中，接下来为人脸剪切部分
        # 用来储存生成的单张人脸的路径
        path_save = "./curve/faces/" #你可以将这里的路径换成你自己的路径
        os.makedirs(path_save,exist_ok=True)
        #剪切图片
        if True:
            print('len(dets)',len(dets))
            print('count face',count_face)
            if count_face>1:
                return 'error','error','error','error'
            else:
                for num, b in enumerate(dets): # dets中包含了人脸框和五个特征点的坐标
                    print('b:',b)
                    if b[4] < self.args['FaceDetector']['vis_thres']:
                        continue
                    b = list(map(int, b))

                    # # landms，在人脸上画出特征点，要是你想保存不显示特征点的人脸图，你可以把这里注释掉
                    # cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                    # cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                    # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                    # cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                    # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

                
                    img_blank = img_raw_source[int(b[1]):int(b[3]), int(b[0]):int(b[2])] # height, width
                        
                    # cv2.namedWindow("img_faces")  # , 2)
                    # cv2.imshow("img_faces", img_blank)  #显示图片
                    # print(img_blank)
                    cv2.imwrite(path_save + "img_face_42" + str(num + 1) + ".jpg", img_blank)  #将图片保存至你指定的文件夹
                    print("Save into:", path_save + "img_face_4" + str(num + 1) + ".jpg")
                    points=[]

                    x1,y1=(b[0],b[1])
                    x2,y2=(b[2],b[3])

                    detect_result=[(x1,y1),(x2,y2)]
                    for i in range(5):
                        x_index=2*i+5
                        y_index=2*i+6
                        # point=(b[x_index]-int(b[0]),b[y_index]-int(b[1]))\
                        point=(b[x_index],b[y_index])
                        print(point)
                        points.append(point)
                    return 'success',img_raw_source,detect_result,points

        '''
        #保存人脸框，特征点的坐标信息到txt中
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        fw = open(os.path.join(args.save_folder, '__dets.txt'), 'w')  #在指定的文件夹中生成并打开一个名为__dets的txt文件
        if args.save_folder:
            fw.write('{:s}\n'.format(img_name)) #在txt中写入图片名字
            for k in range(dets.shape[0]): #遍历dets中的坐标信息，dets中的信息包括（人脸框的左上角x,y坐标，人脸框右下角x,y坐标，人脸检测精度scores，五个特征点的x,y坐标，共15个信息）
                xmin = dets[k, 0]
                ymin = dets[k, 1]
                xmax = dets[k, 2]
                ymax = dets[k, 3]
                score = dets[k, 4]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                landms1_x = dets[k, 5]
                landms1_y = dets[k, 6]
                landms2_x = dets[k, 7]
                landms2_y = dets[k, 8]
                landms3_x = dets[k, 9]
                landms3_y = dets[k, 10]
                landms4_x = dets[k, 11]
                landms4_y = dets[k, 12]
                landms5_x = dets[k, 13]
                landms5_y = dets[k, 14]
                #将人脸框，人脸检测精度，五个特征点的坐标信息写入到txt文件中
                fw.write('{:d} {:d} {:d} {:d} {:.10f} {:d} {:d} {:d} {:d} {:d} {:d} {:d} {:d} {:d} {:d}\n'.format(int(xmin), int(ymin), int(w), int(h), score, int(landms1_x),int(landms1_y),
                                                                                                                  int(landms2_x), int(landms2_y), int(landms3_x), int(landms3_y), int(landms4_x),
                                                                                                                  int(landms4_y), int(landms5_x), int(landms5_y)))
        #写入完毕，关闭txt文件
        fw.close()
        '''

    def aligin(self,image,detect_result,points,scale=1.0):

        (x1,y1),(x2,y2)=detect_result
        shape = image.shape
        height = shape[0]
        width = shape[1]
        imgSize1 = [112,96]
        coord5point1 = [[30.2946, 51.6963],  # 112x96的目标点
               [65.5318, 51.6963],
               [48.0252, 71.7366],
               [33.5493, 92.3655],
               [62.7299, 92.3655]]

        new_x1 = max(int(1.50 * x1 - 0.50 * x2),0)
        new_x2 = min(int(1.50 * x2 - 0.50 * x1),width-1)
        new_y1 = max(int(1.50 * y1 - 0.50 * y2),0)
        new_y2 = min(int(1.50 * y2 - 0.50 * y1),height-1)
        left_eye_x = points[0][0]
        right_eye_x = points[1][0]
        nose_x = points[2][0]
        left_mouth_x = points[3][0]
        right_mouth_x = points[4][0]
        left_eye_y = points[0][1]
        right_eye_y = points[1][1]
        nose_y = points[2][1]
        left_mouth_y = points[3][1]
        right_mouth_y = points[4][1]
        # 得到外扩100%后图中关键点坐标
        new_left_eye_x = left_eye_x - new_x1
        new_right_eye_x = right_eye_x - new_x1
        new_nose_x = nose_x - new_x1
        new_left_mouth_x = left_mouth_x - new_x1
        new_right_mouth_x = right_mouth_x - new_x1
        new_left_eye_y = left_eye_y - new_y1
        new_right_eye_y = right_eye_y - new_y1
        new_nose_y = nose_y - new_y1
        new_left_mouth_y = left_mouth_y - new_y1
        new_right_mouth_y = right_mouth_y - new_y1
        face_landmarks = [[new_left_eye_x,new_left_eye_y], # 在扩大100%人脸图中关键点坐标
                            [new_right_eye_x,new_right_eye_y],
                            [new_nose_x,new_nose_y],
                            [new_left_mouth_x,new_left_mouth_y],
                            [new_right_mouth_x,new_right_mouth_y]]
        face = image[new_y1: new_y2, new_x1: new_x2]
        dst1 = self.warp_im(face,face_landmarks,coord5point1)
        crop_im1 = dst1[0:imgSize1[0],0:imgSize1[1]] #bgr
        return crop_im1[:,:,::-1]  #rgb
    def warp_im(self,img_im, orgi_landmarks,tar_landmarks):
        pts1 = np.float64(np.matrix([[point[0], point[1]] for point in orgi_landmarks]))
        pts2 = np.float64(np.matrix([[point[0], point[1]] for point in tar_landmarks]))
        M = self.transformation_from_points(pts1, pts2)
        dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
        return dst

    def transformation_from_points(self,points1, points2):
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)
        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2
        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2
        U, S, Vt = np.linalg.svd(points1.T * points2)
        R = (U * Vt).T
        return np.vstack([np.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T)),np.matrix([0., 0., 1.])])

if __name__ == '__main__':
    face_detector=FaceDector()
    flag,img,detect_result,points=face_detector.detect('/home/wht/face/code/Pytorch_Retinaface/curve/2.jpg')
 
    i=0
    for (x,y) in points:
        # print(x,y)
        if False:
            cv2.circle(img, (x, y), 1, (0, 0, 255), 4)
        i+=1
        
    #left eye
    #right eye
    #noise
    #left mouth
    #right mouth
    cv2.imwrite('./tmp.jpg', img) 
    if not flag=='error':
        rot_img=face_detector.aligin(img,detect_result,points)
        
        import imageio
        imageio.imwrite('./tmp_1.jpg',rot_img)
       
    # net and model
    

    # testing begin
        