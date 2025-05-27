from vidstab import VideoStabiliser, open_video

video = open_video("test_video.mp4")
stabiliser = VideoStabiliser(focal_length=705, verbose=False)
stabiliser.stabilise(video)
video.save("stabilised_video.mp4")
