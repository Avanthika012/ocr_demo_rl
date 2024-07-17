from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
from models.ocr_model import OCR
import base64
import sys
from OpenSSL import SSL
from flask_cors import CORS
import json
import datetime
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# Get the logger
from logger_setup import get_logger
logger = get_logger(__name__, "ocr.log", console_output=True)

# Importing messages module
from messages import Messagesx
msgs = Messagesx()
logger.info(msgs.welcome("OCR DEMO"))

app = Flask(__name__)
# Enable CORS for specific origin
CORS(app, resources={r"/*": {"origins": "https://192.168.11.53:3000"}})
# Define the paths
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'

# Ensure the folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
logger.info("UPLOAD_FOLDER = 'uploads' & RESULT_FOLDER = 'results' folders created!!!")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

try:
    with open('./model_jsons/paramx.json', 'r') as f:
        params = json.load(f)
    # Initialize the OCR model with the result path
    ocr_modelx = OCR(params,logger=logger,res_path=RESULT_FOLDER)
except:
    print(f"\n [ERROR] {datetime.datetime.now()} OCR model loading failed!!!\n ")
    traceback.print_exception(*sys.exc_info())
    sys.exit(1)

# Define allowed extensions for image files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/ping')
def ping():
    return jsonify({
            'response': "pong"
        })

@app.route('/upload', methods=['POST'])
def upload_file():
    # Log the incoming request headers and content type
    logger.info(f"Request headers: {request.headers}")
    logger.info(f"Request content type: {request.content_type}")

    # Debug the request form and files
    logger.info(f"Request form data: {request.form}")
    logger.info(f"Request files: {request.files}")
    logger.info(f"Request manualEntry: {request.form.get('manualEntry')}")
    manualEntryx = request.form.get('manualEntry')


    # Check if the request has the file part
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    logger.info(f"Received file: {file.filename}")
    
    # If user does not select file, browser may submit an empty part without filename
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the image with OCR
        res_img = process_image(file_path,manualEntryx)
        img_base64 = base64.b64encode(res_img).decode('utf-8')
        
        return jsonify({
            'image': img_base64
        })
    else:
        logger.error("Invalid file type")
        return jsonify({'error': 'Invalid file type'}), 400

def process_image(file_path,manualEntryx):
    # Open the image file using OpenCV
    img = cv2.imread(file_path)
    logger.info(f"Working with img: {file_path}")
    
    # Use the OCR model to process the image
    res_txt, result_img_path = ocr_modelx(img,img_name=file_path.split("/")[-1],manualEntryx=manualEntryx)
    logger.info(f"\nres_txt: {res_txt}\n result_img_path: {result_img_path}\n")
    
    # Read the resulting image as bytes
    with open(result_img_path, 'rb') as f:
        res_img_bytes = f.read()
    
    return res_img_bytes

if __name__ == '__main__':
    context = ('/home/frinksserver/aryan/ocr-demo-frontend-4Jul/ocr-demo-frontend/192.168.11.53.pem', '/home/frinksserver/aryan/ocr-demo-frontend-4Jul/ocr-demo-frontend/192.168.11.53-key.pem')  # Provide the path to your certificate and key files
    logger.info(msgs.processstart)
    app.run(host='0.0.0.0', port=9000, ssl_context=context, debug=True)
    logger.info(msgs.processend)
