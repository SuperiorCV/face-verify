import glob,os
dir='/home/wht/face/datasets/caisa/CASIA-WebFace-112X96'
files=glob.glob(os.path.join(dir,'*','*.jpg'))
print(len(files))
assert len(files)==490871,'WEBFACE 数据集缺失'