import numpy as np
import cv2 as cv
import rawpy
from PIL import Image
import os
import shutil
def read_cr2(file):
    with Image.open(file) as raw:
        rgb = raw.convert("RGB")#raw.postprocess()
        rgb = np.array(rgb)
        #rgb = np.float32(rgb/65535.0*255.0)
        #rgb = np.asarray(rgb, dtype=np.uint8)
        bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
        #cv.imshow("img",bgr)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
        return bgr
def read_arw(file):
    with rawpy.imread(file) as raw:
        rgb = raw.postprocess()
        rgb = np.array(rgb)
        #rgb = np.float32(rgb/65535.0*255.0)
        #rgb = np.asarray(rgb, dtype=np.uint8)
        bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
        #cv.imshow("img",bgr)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
        return bgr
dir_paths = ["./f2.8_r","./f8-16_raw","./f20-22_raw"]

count = 5
base = 1
end = 11
l_name = "L"
r_name = "R"
for dir_path in dir_paths[0:1]:
    all_dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    all_dirs = sorted(all_dirs)
    for dir in all_dirs[0:]:
        imagesL = [img for img in os.listdir(os.path.join(dir_path, dir,l_name)) if img.endswith(".ARW")]
        imagesR = [img for img in os.listdir(os.path.join(dir_path, dir,r_name)) if img.endswith(".ARW")]
        imagesL = sorted(imagesL)
        imagesR = sorted(imagesR)
        print(imagesL , " " , dir)
        print(imagesR , " " , dir)
        for i in range(0,count):
            imgL = read_arw(os.path.join(dir_path, dir, l_name, imagesL[i]))
            imgR = read_arw(os.path.join(dir_path, dir, r_name, imagesR[i]))
            cv.imwrite(f"{dir_path}/{dir}/{l_name}/{imagesL[i][:-4]}.JPG", imgL)
            cv.imwrite(f"{dir_path}/{dir}/{r_name}/{imagesR[i][:-4]}.JPG", imgR)
        print('done one')
    print(f'done {dir_path}')
