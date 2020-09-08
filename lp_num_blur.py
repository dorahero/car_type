import cv2
import numpy as np
import glob

ff = glob.glob('../yolo_data/*/images/*.jpg')
# ff = glob.glob('../test_lp/20200827_101123.jpg')
for f in ff:
#     img = cv2.imread(f, 0)
    img = cv2.imread(f)
#     a = 0
#     for i in img.shape:
#         if i > a:
#             a = i
#     l = 640
#     r = a / 640
    r = 0.02
    img = cv2.resize(img, None, fx=r, fy=r, interpolation=cv2.INTER_NEAREST)
    print(img.shape)
    a = 0
    for i in img.shape:
        if i > a:
            a = i
    l = 640
    r = 640 / a
    img = cv2.resize(img, None, fx=r, fy=r, interpolation=cv2.INTER_NEAREST)
    print(img.shape)
#     ret, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (81, 81), 0)
#     cv2.imwrite('result.jpg', img)
    cv2.imwrite(f, img)

# a = 1.3
# O = I * float(a)
# O[O > 255] = 255
# O = np.round(O)
# O = O.astype(np.uint8)
# cv2.imwrite('O.jpg', O)
# result = np.uint8(np.clip((1.5*I + 100), 0, 255))
# cv2.imwrite('result.jpg', result)

# im = cv2.imread("/home/jovyan/yolov5_new/yolov5/target/lp.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE) 
# cv2.AdaptiveThreshold(im, im, 255, cv2.CV_ADAPTIVE_THRESH_MEAN_C, 
#             cv2.CV_THRESH_BINARY, blockSize=31, param1=15) 
# cv2.imwrite('result.jpg', im)