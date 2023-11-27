# # import cv2
# import streamlit as st
# import numpy as np
# import tempfile
# import cv2
# from cv2 import VideoCapture
# from cv2 import waitKey
# import argparse
# # import cv2 as cv
# # import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# # import tensorflow as tf
# # from streamlit_webrtc import WebRtcMode, webrtc_streamer
# # import av
# # # import pyttsx3  



# # # Use this line to capture video from the webcam
# # def video_frame_callback(frame):
# #     img = frame.to_ndarray(format="bgr24")

# #     flipped = img[::-1,:,:]

# #     return av.VideoFrame.from_ndarray(flipped, format="bgr24")

# # mpHands = mp.solutions.hands
# # hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
# # mpDraw = mp.solutions.drawing_utils

# # model = load_model('mp_hand_gesture')
# # f = open('gesture.names', 'r')
# # classNames = f.read().split('\n')
# # f.close()
# # print(classNames)


# # # Set the title for the Streamlit app
# # cap = cv2.VideoCapture(0)
# # cap = webrtc_streamer(
# #     key="object-detection",
# #     # video_frame_callback=video_frame_callback,
# #     video_frame_callback=video_frame_callback,
# #     # media_stream_constraints={"video": True, "audio": False},
# #     async_processing=True,
# # )
# # # cap = cv2.VideoCapture()
# st.title("SignSense")

# # frame_placeholder = st.empty()

# # # Add a "Stop" button and store its state in a variable
# stop_button_pressed = st.button("Stop")
# # # cap.isOpened() and
# while cap.isOpened() and not stop_button_pressed:
#     ret, frame = cap.read()

#     if not ret:
#         st.write("The video capture has ended.")
#         break

#     ret, frame = cap.read()
#     if not ret:
#         continue

#     x, y, c = frame.shape

# #     # Flip the frame vertically
#     frame = cv.flip(frame, 1)
#     framergb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

# #     # Get hand landmark prediction
# #     result = hands.process(framergb)

# #     # print(result)
    
# #     className = ''

# #     # post process the result
# #     if result.multi_hand_landmarks:
# #         landmarks = []
# #         for handslms in result.multi_hand_landmarks:
# #             for lm in handslms.landmark:
# #                 # print(id, lm)
# #                 lmx = int(lm.x * x)
# #                 lmy = int(lm.y * y)

# #                 landmarks.append([lmx, lmy])

# #             # Drawing landmarks on frames
# #             mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

# #             # Predict gesture
# #             prediction = model.predict([landmarks])
# #             # print(prediction)
# #             classID = np.argmax(prediction)
# #             className = classNames[classID]

# #     # show the prediction on the frame
# #     cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
# #                    1, (0,0,255), 2, cv2.LINE_AA)

# #     # You can process the frame here if needed
# #     # e.g., apply filters, transformations, or object detection
    

# #     # Convert the frame from BGR to RGB format
#     frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

# #     # Display the frame using Streamlit's st.image
#     frame_placeholder.image(frame, channels="RGB")

# #     # Break the loop if the 'q' key is pressed or the user clicks the "Stop" button
#     if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed: 
#         break

# cap.release()

import cv2
import streamlit as st
import numpy as np
import tempfile
import cv2
from cv2 import VideoCapture
from cv2 import waitKey
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow as tf
# from streamlit_webrtc import WebRtcMode, webrtc_streamer

# import pyttsx3 



    
    

# Use this line to capture video from the webcam

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

model = load_model('mp_hand_gesture')
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# Set the title for the Streamlit app
cap = cv2.VideoCapture(-1)
st.write("SignSense 👌")

frame_placeholder = st.empty()

# Add a "Stop" button and store its state in a variable
stop_button_pressed = st.button("Stop")

while cap.isOpened() and not stop_button_pressed:
    ret, frame = cap.read()

    if not ret:
        st.write("The video capture has ended.")
        break

    ret, frame = cap.read()
    if not ret:
        continue

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

    # You can process the frame here if needed
    # e.g., apply filters, transformations, or object detection
    

    # Convert the frame from BGR to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame using Streamlit's st.image
    frame_placeholder.image(frame, channels="RGB")

    # Break the loop if the 'q' key is pressed or the user clicks the "Stop" button
    if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed: 
        break

cap.release()
cv2.destroyAllWindows()



