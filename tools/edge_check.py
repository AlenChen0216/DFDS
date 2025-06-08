import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
CAL_HEAD = "./new_bench2/"
LARGER = "/L"
SMALLER = "/R"
position = 0
head = os.listdir(CAL_HEAD)
head = sorted(head)
l_place = [CAL_HEAD+i+LARGER for i in head[position:]]
r_place = [CAL_HEAD+i+SMALLER for i in head[position:]]
fileL = [sorted(os.listdir(place)) for place in l_place]
fileR = [sorted(os.listdir(place)) for place in r_place]
fileL = [[f for f in files if f.endswith(".JPG")] for files in fileL]
fileR = [[f for f in files if f.endswith(".JPG")] for files in fileR]
imgs = [[top+"/"+file for file in img] for img,top in  zip(fileL,l_place)]
imgR = [[top+"/"+file for file in img] for img,top in  zip(fileR,r_place)]
THR = 40
MAX = 37
IMG = None
def mouse(event,x,y,flags,param):
    global IMG
    if event == cv.EVENT_LBUTTONDOWN:
        print("x : ",x," y : ",y,end=" value : ")
        print(IMG[y,x])

def trackbar(x):
    THR = cv.getTrackbarPos("threshold","edge")
    MAX = cv.getTrackbarPos("maxVal","edge")
    edge = cv.Canny(IMG,THR,MAX)
    cv.imshow("edge",edge)

def lpTracker(x):
    global IMG
    ksize = cv.getTrackbarPos("ksize","lap")
    scale = cv.getTrackbarPos("scale","lap")
    delta = cv.getTrackbarPos("delta","lap")
    lap = cv.Laplacian(IMG,cv.CV_64F,ksize=2*ksize+1,scale=scale,delta=delta)
    lap = cv.convertScaleAbs(lap)
    cv.imshow(f"lap",lap)

def edge_check():
    global imgR,THR,MAX,IMG
    count = 1
    for img_ in imgR:
        img = cv.imread(img_[0])
        img = cv.resize(img, (1824,1216) ,interpolation=cv.INTER_AREA)
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        edge = cv.Canny(img,THR,MAX)
        print(f"count={count}")
        count+=1
        cv.namedWindow(f"edge",cv.WINDOW_NORMAL)
        cv.resizeWindow("edge", (912,608)) 
        cv.imshow("edge",edge)
        IMG = img
        cv.createTrackbar("threshold","edge",THR,255,trackbar)
        cv.createTrackbar("maxVal","edge",MAX,255,trackbar)
        key = cv.waitKey(0)
        if key == 27 or key == ord('q'):
            break
        cv.destroyAllWindows()

def f(x,a,b):
    return a*x**2 + b*x

def line():
    polyline = np.linspace(0,0.004,2000)
    para = [[1.48833016e+07 ,-7.87175883e+03], 
            [1.37866710e+07 ,-7.99882451e+03],
            [1.25208702e+07 ,-5.60481600e+03],
            [24958794.74290193  , -28037.3159782],
            [26076444.1181337  , -29585.93506573],
            [26942840.95145722 , -30469.76259743]]
    kernel = [48,24,12]
    for i in range(len(para)-3):
        plt.plot(polyline,f(polyline,*para[i]),label=f"quadratic line with only edge kernel={kernel[(i%3)]}")
    for i in range(len(para)-3,len(para)):
        plt.plot(polyline,f(polyline,*para[i]),label=f"quadratic line kernel={kernel[(i%3)]}",linestyle="--")
    plt.xlabel("polyline")
    plt.ylabel("distance")
    plt.ylim(0,140)
    plt.legend()
    plt.show()
def lap():
    global IMG,imgs
    cnt = 1
    for img_ in imgs:
        img = cv.imread(img_[0])
        img = cv.resize(img, (1824,1216),interpolation=cv.INTER_AREA)
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img = np.float64(img)
        lap = cv.Laplacian(img,cv.CV_64F,ksize=7,scale=2,delta=0)
        lap_s = lap**2
        IMG = lap_s
        print(lap_s.max(),lap_s.min(),lap_s.mean())
        mask = np.where(lap_s<=8000000.0,0.0,lap_s)
        show_m = np.zeros_like(mask)
        show_m = cv.cvtColor(show_m.astype(np.uint8),cv.COLOR_GRAY2BGR)
        show_m[mask!=0.0] = [0,0,255]
        show_m[mask==0.0] = [255,255,255]
        show_m = show_m.astype(np.uint8)
        
        cv.namedWindow(f"lap",cv.WINDOW_NORMAL)
        cv.resizeWindow(f"lap", (912,608))
        cv.setMouseCallback(f"lap",mouse)
        cv.imshow(f"lap",show_m)
        key = cv.waitKey(0)
        if key == 27 or key == ord('q'):
            cv.destroyAllWindows()
            break
        elif key == ord('s'):
            if os.path.isdir("./lap_edge") == False:
                os.makedirs("./lap_edge")
            cv.imwrite(f"./lap_edge/lap_{cnt}.jpg",show_m)
            print("Save Image")
        cv.destroyAllWindows()
        cnt += 1
if __name__ == "__main__":
    lap()
    #edge_check()
    #line()