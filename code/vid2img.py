import cv2
import os

outdir = 'data/sample_drive/paper/'
cap = cv2.VideoCapture('data/sample_drive/paper.avi')

try:
    if not os.path.exists(outdir):
        os.makedirs(outdir)
except OSError:
    print('Error: cannot create dir of output data.')

id = 0
while(True):
    ret, frame = cap.read()

    fn = outdir + str(id) +'.png'
    # print('Creating...' + fn)
    cv2.imwrite(fn, frame)

    id += 1

cap.release()
