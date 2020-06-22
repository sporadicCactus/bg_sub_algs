import cv2
import numpy as np
import argparse
import os
from vibe import ViBe, ViBePlus

parser = argparse.ArgumentParser()
parser.add_argument("--source", "-s", type=str, help="Path to a source video or a folder with frames.")
parser.add_argument("--alg", "-a", type=str, default="vibe_p", help="Algorithm: vibe or vibe_p")
parser.add_argument("--downscale", "-d", type=int, default=1, help="A factor by which to downscale frames before processing")
args = parser.parse_args()


is_video = os.path.isfile(args.source)

algs = {"vibe": ViBe, "vibe_p": ViBePlus}
segmenter = algs[args.alg]()

def process_frame(frame):
    resized_frame = cv2.resize(frame, (frame.shape[1]//args.downscale, frame.shape[0]//args.downscale))
    seg_map = segmenter.step(resized_frame) # each pixel is 0 or 255
    
    seg_map = seg_map/255
    #smoothing the segmentation map, not strictly necessary
    seg_map = (cv2.boxFilter(seg_map, -1, (5,5)) > .5).astype(float)

    #upscaling to original resolution
    seg_map = cv2.resize(seg_map, (frame.shape[1], frame.shape[0]))


    #masking the image to highlight movement
    seg_map = (0.5 + seg_map)/1.5
    masked_frame = (seg_map[...,None] * frame).astype(np.uint8)

    return masked_frame 


if is_video:
    cap = cv2.VideoCapture(args.source)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            out_frame = process_frame(frame)
            cv2.imshow('press q to close', out_frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    filenames = os.listdir(args.source)

    for filename in filenames:
        frame = cv2.imread(os.path.join(args.source, filename))
        out_frame = process_frame(frame)

        cv2.imshow('press q to close', out_frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
