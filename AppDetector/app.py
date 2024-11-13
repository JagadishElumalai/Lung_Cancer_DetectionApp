from flask import Flask, render_template, request, send_from_directory
import os
from werkzeug.utils import secure_filename
import lung_processing

UPLOAD_FOLDER = 'AppDetector/uploads'
SAVED_FOLDER = '/static/assets/'
ALLOWED_EXTENSIONS = {'mhd'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SAVED_FOLDER'] = SAVED_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/upload",methods=['POST'])
def upload_file():
    if ('mhd' not in request.files) or ('raw' not in request.files):
        return 'No file part'
    mhd= request.files['mhd']
    raw= request.files['raw']
    if mhd.filename == '' or raw.filename == '':
        return 'No selected file'
    if mhd and allowed_file(mhd.filename):
        mhd_filename = secure_filename(mhd.filename)
        raw_filename = secure_filename(raw.filename)
        mhd_filepath = os.path.join(app.config['UPLOAD_FOLDER'],mhd_filename)
        raw_filepath = os.path.join(app.config['UPLOAD_FOLDER'],raw_filename)
        mhd.save(mhd_filepath)
        raw.save(raw_filepath)
        
        results = lung_processing.process_ct_scan(mhd_filepath)

        return render_template('results.html', 
                               normalized_img=results['normalized_img'], 
                               mask_video=results['mask_video'],
                               final_img_bbox_video=results['final_img_bbox_video'],
                               cancer_probabilities=results['cancer_probabilities'])
    else:
        return 'File type not allowed'

if __name__ == '__main__':
    app.run(debug=True)
