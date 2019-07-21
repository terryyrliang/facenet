##################################
### webface interface, obsolete, will use java spring boot to replace
##################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
app_path = os.environ['APP_PATH']
for p in app_path.split(';'):
    sys.path.append(p)
from flask import Flask, request, redirect, jsonify
from configuration import configuration_service as cs
from service import tiny_predict_queue_service as pqs
from utils import utils
from utils import image_utils as iu
import traceback

UPLOAD_FOLDER = cs.image_upload_path
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def build_response(person_name, user_id, index):
    resp = jsonify({'result': person_name, 'user_id' : user_id, 'index' : index})
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers[
        'Access-Control-Allow-Headers'] = 'Origin,X-Requested-With,Content-Type,Accept,Content-Encoding'
    resp.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    return resp

def pre_get_userid():
    return utils.get_random_str()

def get_userid(request, default_userid):
    user_id = request.form["userId"]
    if user_id == None or user_id == '' or len(user_id) == 0:
        user_id = default_userid
    return user_id

def set_userid(resp, user_id):
    if user_id != None or len(user_id) > 0:
        resp.set_cookie("user_id", user_id, max_age=360000)

def create_dir(session_dir):
    if os.path.isdir(session_dir):
        return
    os.mkdir(session_dir)

@app.route('/detect_face', methods=['GET', 'POST', 'OPTIONS'])
def detect_face():
    user_id = pre_get_userid()
    try:
        if request.method == 'POST':
            # check if the post request has the file part
            if 'image' not in request.files:
                print('no file')
                return build_response('test', user_id, '')
            file = request.files['image']
            user_id = get_userid(request, user_id)
            label = request.form['labelName']
            index = request.form['index']
            if pqs.can_enqueue(user_id, label) == False:
                print('Detecting')
                return build_response('detecting', user_id, index)
            image_root_dir = app.config['UPLOAD_FOLDER'] + '/' + user_id
            create_dir(image_root_dir)
            if file.filename == '':
                print('No filename')
                return redirect(request.url)
            predict_dir = image_root_dir + "/" + cs.detect_dir
            create_dir(predict_dir)
            target_file = predict_dir + '/' + utils.get_timebased_imagename() + cs.predict_image_extention
            if (os.path.isfile(target_file) and not utils.is_open(target_file)):
                print('Remove old file')
                os.remove(target_file)
            if not utils.is_open(target_file):
                print('Save image to file')
                file.save(target_file)
            if iu.validate_face_with_path(target_file) == False:
                print('Image not clear')
                os.remove(target_file)
                return build_response('test', user_id, index)
            result = pqs.enqueue_predict(user_id, label, image_root_dir, target_file)
            if result == None:
                print('Detecting and response')
                return build_response('detecting', user_id, index)
            # enqueue to refresh the cache before response
            print('Last detect completed, submit a new one')
            pqs.dequeue_predict(user_id, label, image_root_dir, target_file)
            pqs.enqueue_predict(user_id, label, image_root_dir, target_file)
            return build_response(result.owner, user_id, index)
    except Exception as e:
        traceback.print_exc()
        info = traceback.format_exc()
    return build_response('test', user_id, '0')


if __name__ == "__main__":
    pthread = pqs.PredictThread()
    pthread.start()
    print('Predict Thread start...')
    app.run(debug=False, port=5001)