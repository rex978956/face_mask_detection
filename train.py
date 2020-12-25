from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import numpy as np 
import os

def get_dataset():
    directory = "dataset"
    categories = ["with_mask", "without_mask"] # 標籤 
    data = [] # 儲存圖片集
    labels = [] # 儲存圖片的標籤

    for category in categories:
        path = os.path.join(directory, category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)
            data.append(image)
            labels.append(category)
            
    labels = LabelBinarizer().fit_transform(labels) # Encode 成 sklearn 能用的
    labels = to_categorical(labels)
    data = np.array(data,dtype="float32")
    labels = np.array(labels)
    
    return data, labels

def create_model():
    # 直接使用 kearas 提供的 MobileNetV2 
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape = (224, 224, 3))
    base_model.trainable = False # 固定 MobileNetV2 訓練的參數不變
    model = Sequential(name="face_mask_detector") # 初始化一個 model
    model.add(base_model) # 將 MobileNetV2 加入
    model.add(AveragePooling2D(pool_size=(7, 7))) 
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax"))
    return model


def main():
    data, labels = get_dataset()
    
    # 將 80% 的資料當作是訓練資料，其他的 20% 則做測試資料用
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=13)
    
    batch_size = 32
    
    epochs = 10

    # 增強訓練資料，避免 overfitting
    train_image_generator = ImageDataGenerator(
        rotation_range=45, # 影象隨機旋轉 -45 ~ 45 度
        width_shift_range=0.2, # 水平平移程度
        height_shift_range=0.2, # 垂直平移程度
        shear_range=0.2, # 隨機錯切換角度
        zoom_range=0.2, # 隨機縮放
        horizontal_flip=True, # 水平翻轉
        vertical_flip = True, # 垂直翻轉
        fill_mode="nearest" # 填充空畫素的方法
    )

    train_data_gen = train_image_generator.flow(trainX, trainY, batch_size = batch_size)
    
    model = create_model()
    
    model.summary() # 將 model 結構印出來
    
    # 因為結果只有2種 所以 loss 函數適合使用 binary_crossentropy
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
    
    # 訓練
    history = model.fit(
        train_data_gen,
        steps_per_epoch = len(trainX) // batch_size,
        validation_data =(testX, testY),
        validation_steps = len(testX) // batch_size,
        epochs=epochs
    )
    
    model.save("mask_detector.model", save_format="h5") 

    
if __name__ == '__main__':
    main()