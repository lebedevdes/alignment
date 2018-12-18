import face_alignment
import cv2
import time
import numpy as np

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device='cpu')

input_img = cv2.imread('images/10,08_1_1.jpg')

h, w, c = input_img.shape

print(h, w, c)

bboxlist = []
bboxlist.append([146, 183, 310, 354])
detection = np.array(bboxlist)

start = time.time()
# preds = fa.get_landmarks(input_img)
preds = fa.get_landmarks(input_img, detection)
finish = time.time()

print(finish - start)

for i in preds[0]:
    cv2.circle(input_img, (int(i[0]), int(i[1])), 3, (0, 0, 0), -1)


cv2.imshow("test", input_img)
cv2.waitKey()
cv2.destroyAllWindows()