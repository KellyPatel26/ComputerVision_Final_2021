# Final project for CSCI 1430. 
This is deepfake classification model

## My Environment
- Python:
  - python 3.7.2
- Tensorflow:
  - tensorflow-2.5.0

## Downloading the Preprocessed Kaggle Deepfake Detection Dataset
[Preprocessed Kaggle Deepfake Detection Challenge](https://drive.google.com/drive/u/2/folders/1C7uQ_l2ugXKNmjicrPGedYIChzw1vdaA) \
[Original Kaggle Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge/data)

## For training CNN
```
python main.py
```
## For training LSTM
```
python main.py --type LSTM
```
## For training LSTM-F
```
python main.py --type LSTM-F --load_checkpoint ./checkpoints/{type}/{timestamp}/{CNN hdf5 file}
```
## Continuing training
```
python main.py --type {type} --load_checkpoint ./checkpoints/{type}/{timestamp}/{hdf5 file}
```
## For testing
```
python main.py --type {type} --phase test --load_checkpoint ./checkpoints/{type}/{timestamp}/{hdf5 file}
```

## For Extracting Face Data 
```
pip install opencv-python
python extract_faces.py
```
