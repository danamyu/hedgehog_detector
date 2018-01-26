# create video file from detected images

from moviepy.editor import ImageSequenceClip

clip = ImageSequenceClip("video_output/output", fps=2) #create video clip from all images in directory specified in first param
clip.to_videofile("video_output/output/output.mp4", fps=2) #save video clip file to path