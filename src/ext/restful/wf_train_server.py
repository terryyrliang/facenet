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
from werkzeug.utils import secure_filename
from configuration import configuration_service as cs
from service import train_image_queue_service as tqs
from utils import utils
from utils import image_utils as iu
from flask_cors import CORS

UPLOAD_FOLDER = cs.image_upload_path
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'blob'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app, supports_credentials=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def build_response(result, user_id):
    resp = jsonify({'result': result, 'user_id' : user_id})
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers[
        'Access-Control-Allow-Headers'] = 'Origin,X-Requested-With,Content-Type,Accept,Content-Encoding'
    resp.headers['Access-Control-Expose-Headers'] = 'Authorization'
    resp.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    # set_userid(resp, user_id)
    return resp

def get_userid(request):
    user_id = request.form["userId"]
    if user_id == None or user_id == '' or len(user_id) == 0:
        user_id = utils.get_random_str()
    return user_id

def set_userid(resp, user_id):
    if user_id != None and len(user_id) > 0:
        resp.headers["user_id"] = user_id

def create_dir(session_dir):
    if os.path.isdir(session_dir):
        return
    os.mkdir(session_dir)

def setup_dir(image_root_path, label):
    create_dir(image_root_path)
    create_dir(image_root_path + '/raw')
    create_dir(image_root_path + '/160')
    create_dir(image_root_path + '/raw/' + label)
    create_dir(image_root_path + '/160/' + label)

def delete_unclear_image(image_path):
    if os.path.isfile(image_path):
        os.remove(image_path)

@app.route('/upload_snapshot', methods=['GET', 'POST', 'OPTIONS'])
def upload_snapshot():
    if request.method == 'POST':
        # check if the post request has the file part
        rfiles = request.files
        if 'image' not in rfiles:
            print('no file')
            return redirect(request.url)
        file = request.files['image']
        label = request.form['labelName']
        user_id = get_userid(request)
        image_root_path = app.config['UPLOAD_FOLDER'] + '/' + user_id
        if not tqs.can_enqueue(image_root_path, label):
            return build_response('training', user_id)
        if file.filename == '':
            print('no file')
            return build_response('no file', user_id)
        filename = secure_filename(file.filename)
        content_type = file.content_type
        if allowed_file(filename) or content_type == 'image/png':
            setup_dir(image_root_path, label)
            raw_dir = image_root_path + '/raw/' + label
            target_file = raw_dir  + "/" + utils.get_random_str() + '.png'
            if (os.path.isfile(target_file)):
                os.remove(target_file)
            file.save(target_file)
            if iu.validate_face_with_path(target_file) == False:
                delete_unclear_image(target_file)
                return build_response('image not clear', user_id)
            tqs.enqueue_train(image_root_path, label)
            print('Train task is in queue, get ready to response')
            return build_response('success', user_id)
        else:
            print('not allowed to upload')
            return build_response('not allowed to upload', user_id)
    return build_response('other error', 'processing')

@app.route('/upload_picture', methods=['GET', 'POST', 'OPTIONS'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        rfiles = request.files
        if 'image' not in rfiles:
            print('no file')
            return redirect(request.url)
        file = request.files['image']
        label = request.form['labelName']
        user_id = get_userid(request)
        image_root_path = app.config['UPLOAD_FOLDER'] + '/' + user_id
        if not tqs.can_enqueue(image_root_path, label):
            return build_response('training', user_id)
        if file.filename == '':
            print('no file')
            return build_response('no file', user_id)
        if file:
            filename = secure_filename(file.filename)
            content_type = file.content_type
        if file and allowed_file(filename) or content_type == 'image/png':
            setup_dir(image_root_path, label)
            raw_dir = image_root_path + '/raw/' + label
            target_file = raw_dir  + "/" + filename
            if (os.path.isfile(target_file)):
                os.remove(target_file)
            file.save(target_file)
            if iu.validate_face_with_path(filename) == False:
                print('Cannot identify people')
                return build_response('Cannot identify people', user_id)
            tqs.enqueue_train(image_root_path, label)
            print('Train task is in queue, get ready to response')
            return build_response('success', user_id)
        else:
            print('not allowed to upload')
            return build_response('not allowed to upload', user_id)
    return build_response('other error')

if __name__ == "__main__":
    tthread = tqs.TrainThread()
    tthread.start()
    print('Train Thread start...')
    app.run(debug=False, port=5000)