# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import time
import cv2
import os
import argparse

def detect_handler(frame, face_net, mask_net, confidence_threshold):

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

	face_net.setInput(blob) # 將訓練好的模型 拿來套
	detections = face_net.forward() # 回傳結果

	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
        
		confidence = detections[0, 0, i, 2]
        
		if confidence > confidence_threshold: # 偵測人臉的門檻 預設 0.65
			
            # 將範圍圈出來
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# 把人臉的部分切出來，以利訓練好的口罩模型分析
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype='float32')
		preds = mask_net.predict(faces, batch_size=32) # 進行分析有沒有戴口罩

	return locs, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", 
                    default = False, 
                    help = "path to video, if none then use webcam")
    ap.add_argument("-m", "--model", 
                    default = "mask_detector.model", 
                    help = "path to trained model")
    args = vars(ap.parse_args())
    
    confidence_threshold = 0.65
    
    prototxt = "deploy.prototxt"
    weights = "res10_300x300_ssd_iter_140000.caffemodel"
    face_net = cv2.dnn.readNet(prototxt, weights)
    
    mask_net = load_model(args['model']) # mask_detector.model

    if args['file']:
        cap = cv2.VideoCapture(args['file']) # 也可以讀影片
    else: 
        cap = cv2.VideoCapture(0) # 讀取攝影鏡頭

    while True:
        ret, frame = cap.read() # 讀取畫面，若有畫面 ret 會是 True
        
        if ret == False: # 影片結束
            break
        
        frame = imutils.resize(frame, width=900)

        locs, preds = detect_handler(frame, face_net, mask_net, confidence_threshold)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, without_mask) = pred

            label = "Mask" if mask > without_mask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            label = "{}: {:.2f}%".format(label, max(mask, without_mask) * 100)

            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2) # 寫上是否有口罩
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2) # 將人臉範圍圈出來

        cv2.imshow("Project: face mask detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()