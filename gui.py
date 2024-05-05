import tkinter as tk
from tkinter import filedialog 
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.initializers import Orthogonal
import matplotlib.pyplot as plt

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

video_filepath = False

predicted_text = ""

char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

print(
    f"The vocabulary is: {char_to_num.get_vocabulary()} "
    f"(size ={char_to_num.vocabulary_size()})"
)

def load_video(path): 
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

def load_data(path): 
    # path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    print(file_name)
    # File name splitting for windows
    # file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('data','s1',f'{file_name}.mpg')
    # alignment_path = os.path.join('data','alignments','s1',f'{file_name}.align')
    frames = load_video(video_path) 
    # alignments = load_alignments(alignment_path)
    
    return frames

def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

def predict(frames):
    print(frames.shape)
    model = Sequential()
    model.add(Conv3D(128, 3, input_shape=frames.shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))
    model.compile(optimizer=Adam(), loss=CTCLoss)
    model.load_weights('./models/checkpoint')

    res = model.predict(tf.expand_dims(frames, axis=0))
    decoded = tf.keras.backend.ctc_decode(res, input_length=[75], greedy=True)[0][0].numpy()
    print(decoded)
    res2 = [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded]
    predicted_text = res2[0].numpy().decode('utf-8')
    return predicted_text

def select_video():
    global video_filepath
    video_filepath = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.wmv *.mpg")] 
    )
    if video_filepath:
        file_path_label.config(text=f"Selected File: {video_filepath}", font=("Arial", 18))
        text_output.set("Video file selected.\nClick 'Process Video' to start processing.")
        text_display.insert(tk.END, text_output.get() + "\n")
        text_display.insert(tk.END, video_filepath + " Loaded\n")

def process_video():
    if not video_filepath:
        text_output.set("No video uploaded")  
        text_display.insert(tk.END, text_output.get() + "\n")
        return

    text_display.insert(tk.END, "Processing..." + "\n")
    
    video = load_data(video_filepath)

    text_display.insert(tk.END, "Video Processed Successfully" + "\n")
    pr_out.set(predict(video))
    pr_dis.insert(tk.END, pr_out.get() + "\n")


root = tk.Tk()
root.geometry("800x500")
root.title("Video to Text Interface")

video_filepath_var = tk.StringVar()
file_path_label = tk.Label(root, text="No file selected yet", font=("Arial", 18))
file_path_label.pack(padx=10, pady=10)

select_button = tk.Button(root, text="Select Video File", command=select_video)
select_button.pack(padx=10, pady=10)

text_output = tk.StringVar() 
text_display = tk.Text(root, width=50, height=10)  
text_display.pack()

text_output.set("Activity Log:\n")
text_display.insert(tk.END, text_output.get())

text_display.config()  
text_output.set("Video processing is not yet implemented.\nPlease replace this placeholder.")

process_button = tk.Button(root, text="Process Video", command=process_video)
process_button.pack(padx=10, pady=10)

pr_out = tk.StringVar() 
pr_dis = tk.Text(root, width=50, height=10)  
pr_dis.pack()

pr_out.set("Predicted Text:\n")
pr_dis.insert(tk.END, pr_out.get())

root.mainloop()
