import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2

from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

st.title('MLCrats')
st.header('E-Waste Identification and Classification')

device = 'cpu'
if not hasattr(st, 'Detection'):
    st.model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', _verbose=False)
    
    
    # st.model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', _verbose=False)
    


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # vision processing
        flipped = img[:, ::-1, :]

        # model processing
        im_pil = Image.fromarray(flipped)
        results = st.model(im_pil, size=112)
        bbox_img = np.array(results.render()[0])

        return av.VideoFrame.from_ndarray(bbox_img, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)
