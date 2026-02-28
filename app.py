if option == "Webcam":
    st.subheader("Webcam Live Prediction")


    # We use a simple class to avoid using st.session_state inside the thread
    class VideoProcessor:
        def __init__(self):
            self.frame_count = 0

        def recv(self, frame):
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")

            # Only run YOLO every 10th frame to keep the CPU
            # from crashing/glitching the UI
            if self.frame_count % 10 == 0:
                # Use a very small imgsz for the Cloud CPU
                results = model.predict(img, imgsz=160, conf=0.25, verbose=False)
                annotated_frame = results[0].plot()
                return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

            # Return regular frame for the other 9 frames
            return av.VideoFrame.from_ndarray(img, format="bgr24")


    webrtc_streamer(
        key="yolo-final-bridge",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,  # Use the class factory
        media_stream_constraints={
            "video": {"width": 320, "height": 240, "frameRate": 10},
            "audio": False,
        },
        async_processing=True,
    )