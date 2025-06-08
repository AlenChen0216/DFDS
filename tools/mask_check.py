import numpy as np
import cv2 as cv
import os
from  PIL  import Image

CAL_HEAD = "./train_o2/"
mask = [np.load(f"{CAL_HEAD[:-1]}_mask/T{i}.npy") for i in range(1,11)]

LARGER = "/L"
position = 0
head = os.listdir(CAL_HEAD)
head = sorted(head)
l_place = [CAL_HEAD+i+LARGER for i in head[position:]]
fileL = [sorted(os.listdir(place)) for place in l_place]
fileL = [[f for f in files if f] for files in fileL]
imgs = [[top+"/"+file for file in img] for img,top in  zip(fileL,l_place)]
count = 1
def find_intersection(p1,p2,p3,p4):
    A1 = np.float64(p2[1] - p1[1])
    B1 = np.float64(p1[0] - p2[0])
    C1 = A1 * np.float64(p1[0]) + B1 * np.float64(p1[1])

    A2 = np.float64(p4[1] - p3[1])
    B2 = np.float64(p3[0] - p4[0])
    C2 = A2 * np.float64(p3[0]) + B2 * np.float64(p3[1])

    determinant = A1 * B2 - A2 * B1

    if determinant == 0:
        return None  # 平行或重合
    else:
        x = (B2 * C1 - B1 * C2) / determinant
        y = (A1 * C2 - A2 * C1) / determinant
        return int(x), int(y)
def click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x,y)
def read_cr2(file):
    with Image.open(file) as raw:
        rgb = raw.convert("RGB")
        rgb = np.array(rgb)
        bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
        return bgr
drawing = False  # 是否正在绘制
ix, iy = -1, -1  # 起始点坐标

def draw_line(event, x, y, flags, param):
    """
    鼠标事件函数，用于绘制线条并修改掩膜的值。
    """
    global drawing, ix, iy, m

    if event == cv.EVENT_LBUTTONDOWN:  # 鼠标左键按下
        drawing = True
        ix, iy = x, y

    elif event == cv.EVENT_MOUSEMOVE:  # 鼠标移动
        if drawing:
            cv.line(m, (ix, iy), (x, y), [0,0,255], thickness=10)  # 在掩膜上绘制线条
            ix, iy = x, y

    elif event == cv.EVENT_LBUTTONUP:  # 鼠标左键释放
        drawing = False
        cv.line(m, (ix, iy), (x, y), [0,0,255], thickness=10)  # 在掩膜上绘制线条


for m,file in zip(mask,imgs):
    print(np.unique(m)," T ",count," ",np.sum(m))
    m = cv.normalize(m, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    #m = cv.cvtColor(m,cv.COLOR_BGR2GRAY)
    """
    m = cv.cvtColor(m,cv.COLOR_GRAY2BGR)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i,j][0] == 1:
                m[i,j][0] = 0
                m[i,j][1] = 255
                m[i,j][2] = 0
            elif m[i,j][0] == 2.0:
                m[i,j][0] = 0
                m[i,j][1] = 0
                m[i,j][2] = 255
                
    """
    # m = cv.cvtColor(m,cv.COLOR_GRAY2BGR)
    # m[:,:,1] = 0
    # m[:,:,2] = 0
    cv.namedWindow(f"mask_{count}",cv.WINDOW_NORMAL)
    cv.setMouseCallback(f"mask_{count}", draw_line)
    cv.resizeWindow(f"mask_{count}", (912,608))


    # m = cv.erode(m, np.ones((5,5),np.uint8), iterations = 5)
    # m = cv.dilate(m, np.ones((5,5),np.uint8), iterations = 5)

    # m = cv.dilate(m, np.ones((6,6),np.uint8), iterations = 4)
    # m = cv.erode(m, np.ones((6,6),np.uint8), iterations =3)

    #m = cv.erode(m, np.ones((4,4),np.uint8), iterations = 8)
    #m = cv.dilate(m, np.ones((4,4),np.uint8), iterations = 8)
    
    #m = cv.dilate(m, np.ones((5,5),np.uint8), iterations = 7)
    #m = cv.erode(m, np.ones((5,5),np.uint8), iterations = 7)

    
    """
    img = cv.imread(file[0])
    temp_m = cv.resize(img,(1824,1216))
    temp_m = cv.addWeighted(temp_m,0.7,cv.cvtColor(m,cv.COLOR_GRAY2BGR),0.5,0)
    corner,_ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(temp_m, corner, -1, (0,0,255), 3)
    
    right_top = max(corner[0], key=lambda x: x[0][0] - x[0][1])
    right_bottom = max(corner[0], key=lambda x: x[0][0] + x[0][1])
    left_top = min(corner[0], key=lambda x: x[0][0] + x[0][1])
    left_bottom = min(corner[0], key=lambda x: x[0][0] - x[0][1])
    cv.line(temp_m, tuple(right_top[0]), tuple(left_bottom[0]), (0,255,0), 3)
    cv.line(temp_m, tuple(right_bottom[0]), tuple(left_top[0]), (0,255,0), 3)
    center = find_intersection(right_top[0],left_bottom[0],right_bottom[0],left_top[0])
    print(center)
    cv.circle(temp_m, tuple(right_top[0]), 10, (0,255,255), -1)
    cv.circle(temp_m, tuple(right_bottom[0]), 10, (0,255,0), -1)
    cv.circle(temp_m, tuple(left_top[0]),10, (0,255,0), -1)
    cv.circle(temp_m, tuple(left_bottom[0]), 10, (0,255,0), -1)
    cv.circle(temp_m, center, 10, (0,0,255), -1)
    """
    img = cv.imread(file[0]) if file[0].endswith(".JPG") else read_cr2(file[0])
    img_t = cv.resize(img,(1824,1216) )  #(1824,1216)
    temp_m = cv.addWeighted(cv.cvtColor(m,cv.COLOR_GRAY2BGR),0.5,img_t,0.5,0)
    # temp_m = cv.addWeighted(m,0.5,img_t,0.5,0)
    #temp_m = m.copy()
    cv.imshow(f"mask_{count}", temp_m)
    """
    flag = False
    while True:
        temp_m = cv.addWeighted(m,0.5,img_t,0.5,0)
        cv.imshow(f"mask_{count}", temp_m)
        k = cv.waitKey(1) & 0xFF
        if k == ord('m'):
            m = m/255.0
            result = np.where(m[:,:,1] == 1,1,0)
            result = np.where(m[:,:,2] == 1,2,result)
            result = result.astype(np.uint8)
            print(result.shape)
            print(np.unique(result))
            cv.namedWindow("mask",cv.WINDOW_NORMAL)
            cv.resizeWindow("mask", (912,608))
            cv.imshow("mask",result*255)
            k2 = cv.waitKey(0)
            if k2 == ord('q'):
                cv.destroyAllWindows()
                break
            cv.destroyAllWindows()
            np.save(f"{CAL_HEAD[:-1]}_mask/T{count}.npy",result)
            break
        elif k == ord('q'):
            flag = True
            cv.destroyAllWindows()
            break
        elif k == ord('n'):
            cv.destroyAllWindows()
            break
    if flag:
        break
    count += 1
    
    """
    k = cv.waitKey(0)
    if k == ord('s'):
        print("eeeee")
        cv.imwrite(f"T{count}.png",m)
    elif k == ord('q'):
        cv.destroyAllWindows()
        break
    elif k == ord('m'):
        m = m//255
        m.astype(np.uint8)
        np.save(f"{CAL_HEAD[:-1]}_mask/T{count}.npy",m)
    elif k == ord('c'):
        m = m/255.0
        print(np.unique(m))
    count += 1
    cv.destroyAllWindows()

"""
img1 = cv.imread("./bench/T03/smallD/DSC06210.JPG")
img2 = cv.imread("./bench/T03/smallD/DSC06212.JPG")


img1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

print(np.mean(img1)," ",np.mean(img2))

img = np.zeros_like(img1)

print(img.shape)
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        temp = img1[i,j] - img2[i,j]
        if temp <=0 :
            img[i,j] = 0
        else:
            img[i,j] = img1[i,j] - img2[i,j]

print(img.max())

cv.imshow("dif",img)
cv.waitKey(0)
cv.destroyAllWindows()

a = np.load("./F_mask/T1_ch.npy")
b = np.load("./F_mask/T3.npy")
print(np.unique(b))
m = cv.normalize(b, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
print(np.unique(m))
cv.imshow("mask", m)
cv.waitKey(0)
cv.destroyAllWindows()"""
