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



def main(image_dir,custom_name,out_path ="../results"):

    ### creating output paths
    out_img_path = f"{out_path}/img_out/{custom_name}"
    os.makedirs(out_img_path, exist_ok=True)
    logger.info(f"output paths created.")
    ocr_modelx = OCR(res_path=out_img_path)

    logger.info(f"--------- IMAGE INFERENCING STARTED --------- \n")



    logger.info(f"looping through all images.")
    ### reading images and inferencing
    for im_name in tqdm(os.listdir(image_dir)):
        print(f"---------- working with: {im_name}")
        img_path = os.path.join(image_dir, im_name)
        img = cv2.imread(img_path)
        st = time.time()
        res_txt,img_save_name = ocr_modelx(img,img_name =im_name[:-4])
        logger.info(f" time take for OCR model [detection + recognition]: {time.time() - st} \nresult img saved at: {img_save_name}")

    logger.info(f"--- IMAGE INFERENCING COMPLETED ---")

if __name__ == "__main__":
    main(image_dir="../test_img",custom_name="honda_demo2")

