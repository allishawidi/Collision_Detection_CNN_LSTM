import os, random
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import ops
#from tensorflow.keras.metrics import Precision, Recall
import cv2
import numpy as np
import json
import shutil
from collections import deque, Counter
from model_architecture import build_tools
import utils
import config as conf
#import gpu

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
print(os.getenv("TF_GPU_ALLOCATOR"))

ops.reset_default_graph()
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

#gpu.set_memory_limit(1*1024) #change this
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # Set memory limit for the first GPU
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
        )
        print(f"Memory limit set to 1024MB for GPU: {gpus[0]}")
    except RuntimeError as e:
        print(e)

if conf.mode == "train":
    os.makedirs(conf.model_save_folder, exist_ok=True)
    os.makedirs(conf.tensorboard_save_folder, exist_ok=True)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
  filepath=conf.checkpoint_path,
  verbose=1,
  save_weights_only=True,
  save_freq="epoch",
  period=5,
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
  log_dir=conf.tensorboard_save_folder,
  histogram_freq=1,
  write_graph=True,
  write_images=True,
)

def check_data_balance(data_folder):
    labels = []
    for file in os.listdir(data_folder):
        data = np.load(os.path.join(data_folder, file))
        labels.extend(data['name2'].tolist())  # Convert ndarray to list

    label_counter = Counter(map(tuple, labels))  # Convert each ndarray to tuple
    print("Data Balance:", label_counter)

check_data_balance(conf.train_folder)
check_data_balance(conf.valid_folder)
check_data_balance(conf.test_folder)

def _trainer(network, train_generator, val_generator):
  network.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "precision", "recall"])
  network.save_weights(conf.checkpoint_path.format(epoch=0))
  history = network.fit(
      train_generator,
      epochs=conf.epochs,
      steps_per_epoch=len(os.listdir(conf.train_folder)) // conf.batch_size,
      validation_data=val_generator,
      validation_steps=1,
      callbacks=[cp_callback, tensorboard_callback],
  )
  with open(
      os.path.join(conf.base_folder, "files", conf.model_name, "training_logs.json"), "w"
  ) as w:
      json.dump(history.history, w)
 
def inference(network, video_file, output_file):
    print("Starting inference...")
    image_seq = deque([], 8)
    cap = cv2.VideoCapture(video_file)
    counter = 0
    stat = 'AMAN'
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_file, fourcc, 30.0, (800, 600))
    
    if not out.isOpened():
        print("Error: Could not open video writer.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (800, 600))
        frame_resized_small = cv2.resize(frame, (conf.width, conf.height))
        image_seq.append(frame_resized_small)
        
        if counter % 2 == 0 and len(image_seq) == 8:
            np_image_seqs = np.reshape(
                np.array(image_seq) / 255,
                (1, conf.time, conf.height, conf.width, conf.color_channels),
            )
            r = network.predict(np_image_seqs)
            stat = ["AMAN", "TIDAK AMAN"][np.argmax(r, axis=1)[0]]

        text = f"{stat}"
        font = cv2.FONT_HERSHEY_TRIPLEX 
        font_scale = 2  
        thickness = 2
        color = (0, 255, 0) if stat == "AMAN" else (0, 0, 255)
        
        frame_height, frame_width = frame_resized.shape[:2]
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = (frame_width - text_width) // 2
        y = frame_height - 50
        
        cv2.putText(
            frame_resized,
            text,
            (x,y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        out.write(frame_resized)
        counter += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Inference completed. Output video saved to:", output_file)


if __name__ == "__main__":
    try:
        model_tools = build_tools()
        network = model_tools.create_network(conf.model_name)

        if conf.mode == "train":
            train_generator = utils.data_tools(conf.train_folder, "train")
            valid_generator = utils.data_tools(conf.valid_folder, "valid")
            _trainer(
                network, train_generator.batch_dispatch(), valid_generator.batch_dispatch()
            )

        elif conf.mode == "test":
            network.load_weights(os.path.join(conf.model_save_folder, "model_weights_0100.ckpt")).expect_partial()
            output_file = os.path.join(conf.base_folder, "files", "output12.mp4")
            inference(network, os.path.join(conf.base_folder, "files", "input7.mp4"), output_file)

        else:
            p = os.path.join(conf.train_folder, '96.npz')
            np_data = np.load(p, "r")
            imgs = np_data['name1']
            np_image_seqs = np.reshape(
                            np.array(imgs[0]) / 255,
                            (1, conf.time, conf.height, conf.width, conf.color_channels),
                        )
            r = network.predict(np_image_seqs)
            print(np.argmax(r, 1))

    except Exception as e:
        print(f"Main execution failed: {e}")

        
