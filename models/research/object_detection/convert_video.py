# Convert the video to images and store images in video_output directory
import cv2

# create video capture object - enter video title
vc = cv2.VideoCapture("Cat_and_hedgehog.mp4")

while True:
    x = 1

    # make sure video capture object is successfully initialized
    if vc.isOpened():
        ret, frame = vc.read()

    else: ret = False

    # read and capture each frame in video and save image to video_output directory
    while ret:
        ret, frame = vc.read()
        cv2.imwrite('video_output/' + str(x).zfill(7) + '.jpg', frame)
        x += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vc.release()