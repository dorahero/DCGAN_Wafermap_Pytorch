import cv2
import glob
import os


fff = glob.glob("scractch_new/original/*.png")

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
 
    if center is None:
        center = (w / 2, h / 2)
 
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
 
    return rotated

rotate_degree = [90, 180, 270]

for ff in fff:
    image_name = os.path.basename(ff).split(".")[0]
    img = cv2.imread(ff, 0)
    for degree in rotate_degree:
        image = rotate(img, degree)
        cv2.imwrite(f"scractch_new/data_augmentation/{image_name}_{degree}.png", image)
