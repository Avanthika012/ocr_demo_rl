
import torch
import time 
import cv2
import datetime
import os 
import sys

from PIL import Image, ImageDraw, ImageFont
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from models.fasterrcnn_inference import FasterRCNN
from models.paddleocr.tools.infer.predict_rec import PaddleOCRx

# from models.create_colors import Colors  
# colors = Colors()  # create instance for 'from utils.plots import colors'

### ocr code 
class OCR():

    def __init__(self,params,logger,res_path="../results"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        ### loading the model
        if params["use_model"] == "fasterrcnn":
            self.model = FasterRCNN(model_weights=params["models"]["fasterrcnn"]["model_weights"], classes=params["classes"], device=self.device, detection_thr=params["models"]["fasterrcnn"]["det_th"])
            self.det_th = params["models"]["fasterrcnn"]["det_th"]
            print(f"FasterRCNN model created!!!")

        else:
            self.model = None
        if params["use_ocr_model"] == "paddleocr":
            print(f"__init__ OCR: initiating PaddleOCRx")
            self.ocr_model = PaddleOCRx(model_weights=params["ocr_models"]["paddleocr"]["model_weights"])
            print(f"PaddleOCRx model created for text RECOG task!!!")

        else:
            self.ocr_model = None

        self.drop_score = 0.5
        self.logger = logger
        
        self.draw_img_save_dir =  res_path
        os.makedirs(self.draw_img_save_dir, exist_ok=True)


    def __call__(self,image,img_name=None, manualEntryx=None):
        st = time.time()
        ### -------- TEXT DETECTION --------
        boxes, class_names, scores = self.model(image)
        # print(f"[INFO] {datetime.datetime.now()}: time taken for text detection {time.time() - st } seconds")
        self.logger.info(f"[INFO] time taken for text detection {time.time() - st } seconds x {len(class_names)} no. of texts detected!!!")
        detected_texts = []
        detection_scores = []
        detected_bboxes = []

        ### looping through all detected BBoxes or texts on an image
        for i in range(len(class_names)): 
            if scores[i]>=self.det_th: 
                x1,y1,x2,y2 = boxes[i]
                cname= class_names[i]
                detected_bboxes.append([x1,y1,x2,y2 ])

                #### --------    OCR WORK    ----------------
                if self.ocr_model !=None:
                    cropped_image = image[y1:y2, x1:x2]

                    st = time.time() 
                    ocr_text,score = self.ocr_model(cropped_image)
                    detected_texts.append(ocr_text)
                    detection_scores.append(score)
                    
                    # print(f"[INFO] {datetime.datetime.now()}: time taken for text recognition {time.time() - st }  seconds")
                    self.logger.info(f"[INFO] time taken for text recognition {time.time() - st } seconds x detected texts: {detected_texts} detection_scores:{detection_scores}")

        ### plotting results and saving images 
        draw_img = self.draw_ocr_box_txt(
            image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
            boxes=detected_bboxes,
            txts=detected_texts,
            scores=detection_scores,
            drop_score=self.drop_score,
            manualEntryx=manualEntryx
        )
        ### saving output image
        img_save_name = os.path.join(self.draw_img_save_dir, img_name[:-4] if img_name != None else str(self.img_count))+".png"
        cv2.imwrite(
            img_save_name,
            draw_img[:, :, ::-1],
        )
        self.logger.debug(
            "The visualized image saved in {}".format(
                img_save_name
            )
        )

        
        return detected_texts,img_save_name
    

    def draw_ocr_box_txt(self,
        image,
        boxes,
        txts=None,
        scores=None,
        drop_score=0.5,
        font_path="./models/paddleocr/doc/fonts/simfang.ttf",
        font_size_factor=5,
        manualEntryx=None
    ):
        # Validate the font_size_factor
        if not (0.5 <= font_size_factor <= 10.0):
            raise ValueError("font_size_factor should be between 0.5 and 10.0")

        h, w = image.height, image.width
        font = ImageFont.truetype(font_path, int(20 * font_size_factor))

        # Filter out texts based on scores
        valid_texts = [txt for idx, txt in enumerate(txts) if scores is None or scores[idx] >= drop_score]

        img_show = image.copy()

        if manualEntryx is not None:
            manual_entries = manualEntryx.strip().lower()  # Normalize to lower case
        else:
            manual_entries = None

        print(f"Drawing on canvas : valid_texts:{valid_texts}")

        pass_status = "Failed"
        status_color = (255, 0, 0)  # Red for failed

        if manual_entries is not None:
            for text in valid_texts:
                normalized_text = text.strip().lower()
                if any(char in manual_entries for char in normalized_text):
                    pass_status = "Passed"
                    status_color = (0, 255, 0)  # Green for passed
                    break

        # Calculate new height for the status
        line_height = 100  # Height of the line with some margin
        new_h = img_show.height + line_height + 40  # Add some margin at the top
        new_img = Image.new("RGB", (w, new_h), (0, 0, 0))  # Set background color to black
        new_img.paste(img_show, (0, 0))

        draw = ImageDraw.Draw(new_img)
        y_offset = img_show.height + 20  # Start drawing text below the current image with some margin

        # Draw the status text with color
        draw.text((10, y_offset), pass_status, fill=status_color, font=font)

        img_show = new_img  # Update the image with the newly created image

        return np.array(img_show)





