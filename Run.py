from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time
 
from datetime import timedelta
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
 
app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)
 
 
# @app.route('/upload', methods=['POST', 'GET'])
@app.route('/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
 
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
 
        user_input = request.form.get("name")
 
        basepath = os.path.dirname(__file__) 
        print(basepath)
 
        upload_path = os.path.join(basepath, 'static\\images', secure_filename(f.filename))
        print(f.filename)
        f.save(upload_path)
 
        print(upload_path)
        img = cv2.imread(upload_path)
        savepath = os.path.join(basepath, 'static\\images', 'tmp.jpg')
        cv2.imwrite(savepath, img)

        exe1 = "python task1\\SSD_Method\\src\\detect.py --path " + upload_path + " --name " + f.filename
        print(exe1)

        os.system(exe1)

        exe2 = "python task2\\main.py " + "--name " + f.filename
        print(exe2)
        os.system(exe2)

        exe3 = "python task3\\src\\test.py " + "--name " + f.filename
        print(exe3)
        os.system(exe3)

        jsonname = "results\\" + f.filename.split('.')[0]+ '.json'
        fout = open(jsonname, 'r')
        fout.readline()
        v1 = fout.readline()
        v2 = fout.readline()
        v3 = fout.readline()
        v4 = fout.readline()
 
        return render_template('upload_ok.html',userinput=user_input,val1=time.time(), L1 = v1, L2 = v2, L3 = v3, L4 = v4)
 
    return render_template('upload.html')
 
 
if __name__ == '__main__':
    #app.debug = True
    app.run(host='127.0.0.1', port=5000)

