import face_alignment
from skimage import io
import cv2
import time
import pickle
import face_preprocess
import numpy as np

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

m = fa.face_alignment_net.eval()
print(m)


input_image = cv2.imread('images/Tom_Hanks_54745.png')

# start = time.time()
# preds = fa.get_landmarks(input_image)
# end = time.time()

# with open('landmarks.pickle', 'wb') as f:
#     pickle.dump(preds, f)

with open('landmarks.pickle', 'rb') as f:
    preds = pickle.load(f)

# for i in preds[0]:
#     cv2.circle(input_image, (int(i[0]), int(i[1])), 3, (0, 0, 0), -1)

nose = preds[0][30]
right_eye = preds[0][36]
left_eye = preds[0][45]
right_month = preds[0][48]
left_month = preds[0][54]
# chin = preds[0][8]

# p = preds[0][8]

# landmarks = [right_eye, left_eye, nose, right_month, left_month, chin]
landmarks = np.array([right_eye, left_eye, nose, right_month, left_month])
for i in landmarks:
    print(i)
    cv2.circle(input_image, (int(i[0]), int(i[1])), 3, (0, 0, 255), -1)

# print(end - start)


# cv2.imshow("test", input_image)
# cv2.waitKey()
# cv2.destroyAllWindows()

# print(preds)

alignmented = face_preprocess.preprocess(input_image, landmark=landmarks, image_size='112,112')

print(alignmented.shape)

cv2.imshow("test2", alignmented)
cv2.waitKey()
cv2.destroyAllWindows()