
# Videos.py
This python script goes through the videos and does the following:
1. Extract 10 frames from each videos
2. Resize them all to be of 224x224x3 dimensions
    - If the video is horizontal, we pad the bottom
    - If the video is vertical, we pad the right
3. Sorts them into REAL and FAKE folders
