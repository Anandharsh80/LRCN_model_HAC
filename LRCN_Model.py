import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cv2
from moviepy.editor import VideoFileClip


FPS = 60
NUM_FRAMES = 10 * FPS
IMG_SIZE = 224
NUM_CHANNELS = 3
NUM_CLASSES = 1


cnn_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', pooling='avg', input_shape=(IMG_SIZE,IMG_SIZE,NUM_CHANNELS))

for layer in cnn_model.layers[:-15]:
    layer.trainable = False


model = tf.keras.models.Sequential()  

model.add(tf.keras.layers.TimeDistributed(cnn_model, input_shape= (NUM_FRAMES,IMG_SIZE,IMG_SIZE,NUM_CHANNELS)))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
model.add(tf.keras.layers.LSTM(256, activation='relu', return_sequences=False))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

model.compile('adam', loss='categorical_crossentropy')

##################################################################################
path = 'Path relative to location of our script'
##################################################################################

input_list = []
output_list = []

for category in os.listdir(path)[:1]:

    category_folder = path + '/' + category

    for video in os.listdir(category_folder):

        video_path = category_folder + '/' + video
        NUM_CLIPS = int(VideoFileClip(video_path).end/10)

        clip_cnt = 0
        frame_cnt = 0

        vid_frames  = []
        vid_cap = cv2.VideoCapture(video_path)
        success,img = vid_cap.read()
        frame_cnt = frame_cnt + 1

        while success:
            
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            vid_frames.append(img)

            if frame_cnt%(10*FPS) == 0 :
                
                if clip_cnt == NUM_CLIPS:
                    break
                
                clip_cnt = clip_cnt + 1
                vid_frames = np.array(vid_frames)
                input_list.append(vid_frames)
                vid_frames = []
                output_list.append(np.array([1]))

            
            success,img = vid_cap.read()
            frame_cnt = frame_cnt + 1

for category in os.listdir(path)[1:]:

    category_folder = path + '/' + category

    for video in os.listdir(category_folder)[:2]:

        video_path = category_folder + '/' + video
        NUM_CLIPS = int(VideoFileClip(video_path).end/10)

        clip_cnt = 0
        frame_cnt = 0

        vid_frames  = []
        vid_cap = cv2.VideoCapture(video_path)
        success,img = vid_cap.read()
        frame_cnt = frame_cnt + 1

        while success:
            
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            vid_frames.append(img)

            if frame_cnt%(10*FPS) == 0 :
                
                if clip_cnt == NUM_CLIPS:
                    break
                
                clip_cnt = clip_cnt + 1
                vid_frames = np.array(vid_frames)
                input_list.append(vid_frames)
                vid_frames = []
                output_list.append(np.array([0]))

            
            success,img = vid_cap.read()
            frame_cnt = frame_cnt + 1

input_list = np.array(input_list)
output_list = np.array(output_list)

X_train, X_test, y_train, y_test = train_test_split(input_list, output_list, test_size=0.05, random_state=101)

model.fit(X_train, y_train, batch_size=32, epochs= 10, verbose=1, validation_data=(X_test, y_test))

model_save_name = "LRCN_Binary_classifier.model"


####################################################################

save_model_path = "path for saving our model/{}".format(model_save_name)

####################################################################

tf.keras.models.save_model(model, save_model_path, save_format = 'h5')

pred = model.predict(X_test)
class_rep = classification_report(y_test.argmax(axis=1), pred.argmax(axis=1)) 
class_rep = str(class_rep)

cr_file_name = "LRCN_Binary_classifier_cr"

###################################################################

save_cr_path = "path for saving our file/{}".format(cr_file_name)

###################################################################

file_cr = open(save_cr_path, "w")
file_cr.write(class_rep)
file_cr.close()