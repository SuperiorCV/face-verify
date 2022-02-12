from flask import request, Flask, jsonify
from face_main import Face_Modal    
import json
import os
__dir__ = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
modal=Face_Modal()

@app.route("/face/register",methods=['POST'])
def face_register():
    result={}

    id = request.form['id']
    file = request.files.get('img')
    if file is None:
        result['flag']='1'
        result['log']="未上传文件"
        return json.dumps(result,ensure_ascii=False)
    # 直接使用上传的文件对象保存
    else:
        print(os.path.join(__dir__,"./input_data/tmp.jpg"))
        save_path=os.path.join(__dir__,"./input_data/tmp.jpg")
        file.save(save_path)
        flag,log=modal.register_face_check(id=id,img_path=save_path)
        result['flag']=flag
        result['log']=log
        return json.dumps(result,ensure_ascii=False)

@app.route("/face/verify",methods=['POST'])
def face_verify():
    return "Hello moco!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)