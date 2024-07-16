import time 
import cv2
import os 
import sys
from tqdm import tqdm
# sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))


from tools.infer.predict_system import PaddleOCRxDETxRec

# Get the logger
from logger_setup import get_logger
logger = get_logger(__name__, "ocr.log", console_output=True)

# from models.ocr.create_colors import Colors  
# colors = Colors()  # create instance for 'from utils.plots import colors'

### ocr code 
class OCR():

    def __init__(self,res_path = "../results"):

        self.ocr_model = PaddleOCRxDETxRec(rec_model_dir="./model_weights/test_recognition",
                                           det_model_dir = "./model_weights/text_detection",
                                           res_path = res_path
                                           )
        logger.info(f"PaddleOCRxDETxRec model created!!!")

    def __call__(self,image,img_name=None,manualEntryx=None):
        st = time.time()

        res_img,res_txt,img_save_name = self.ocr_model(image,img_name=img_name,manualEntryx=manualEntryx)
        return res_img,res_txt,img_save_name 


