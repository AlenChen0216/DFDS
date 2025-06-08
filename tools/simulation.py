import numpy as np
from skimage.draw import disk
import matplotlib.pyplot as plt
import cv2 as cv
from matplotlib.animation import FuncAnimation, FFMpegWriter
import time
DISTANCE = 50  # distance in pixel
radius = 20
SIG = 0.0

fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10, 8))
mean_text = None
radius_text = None

def update(frame):
    global SIG, mean_text, radius_text
    SIG += 0.01
    grid_size = 4
    img = np.zeros((200, 200), dtype=np.uint8)
    
    # Define the center points of the two circles
    height = 2 * radius + (grid_size - 1) * DISTANCE
    width  = 2 * radius + (grid_size - 1) * DISTANCE
    
    for i in range(grid_size):
        for j in range(grid_size):
            center = (radius + i * DISTANCE, radius + j * DISTANCE)
            rr, cc = disk(center, radius, shape=img.shape)
            img[rr, cc] = 255
            center_cv = (center[1], center[0])  # OpenCV uses (x, y) format
            pt1 = (center_cv[0] - radius, center_cv[1]) 
            pt2 = (200, center_cv[1])
            cv.line(img, pt1, pt2, 255, 1)



    img2 = img.copy()
    
    kernel1 = cv.getGaussianKernel(5, SIG * 1.69230769)
    kernel2 = cv.getGaussianKernel(5, SIG)
    
    kernel1 = kernel1 * kernel1.T
    kernel2 = kernel2 * kernel2.T
    
    img = cv.filter2D(img, -1, kernel1)
    img2 = cv.filter2D(img2, -1, kernel2)
    
    img_temp = img.copy().astype(np.float64)
    img2_temp = img2.copy().astype(np.float64)
    
    dif = (img_temp - img2_temp) ** 2
    mean_value = np.mean(dif)
    
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Filtered Image 1')
    
    ax2.imshow(img2, cmap='gray')
    ax2.set_title('Filtered Image 2')

    ax3.imshow(dif, cmap='gray')
    ax3.set_title('Difference Image')
    if mean_text is not None:
        mean_text.remove()
    if radius_text is not None:
        radius_text.remove()
    radius_text = fig.text(0.5, 0.95, f'Gaussian sigma: {SIG:.2f}', ha='center', va="center", fontsize=16)
    mean_text = fig.text(0.5, 0.9, f'Mean: {mean_value:.2f}', ha='center',va="center",fontsize = 16)
    return ax1, ax2

ani = FuncAnimation(fig, update, frames=range(100), repeat=False,interval=300)
plt.tight_layout()

writer = FFMpegWriter(fps=1000/100)
ani.save('simulation.mp4',writer=writer, dpi=200)
#plt.show()