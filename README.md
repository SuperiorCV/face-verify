

python 3.7
torch 1
```
conda create -n face python=3.7
conda activate face
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install scipy
pip install opencv-python
pip install tqdm
pip install flask
pip install retinaface
pip install PyYaml
pip install imageio
```
/face/register
人脸注册
post:
img
id
返回 注册成功，注册失败（未检测到单个人脸，已注册）

/face/verification
post
img
id

返回 true false
