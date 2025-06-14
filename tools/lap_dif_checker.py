import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
def sort_key(x):
    return int(x.split("/")[0][1:-4])

CAL_HEAD = "./datas/train_w/"


detail = CAL_HEAD[:-1]+"_detail/"
two_folder = os.listdir(detail)
two_folder = sorted(two_folder)
Difference = [detail+two_folder[0]+"/"+name for name in sorted(os.listdir(detail+two_folder[0]),key=sort_key)]
Laplacian = [detail+two_folder[1]+"/"+name for name in sorted(os.listdir(detail+two_folder[1]),key=sort_key)]
print(detail)
size = 10
group_diff = np.array([Difference[i] for i in range(len(Difference))])
group_diff = group_diff.reshape(size,25)
group_lap = np.array([Laplacian[i] for i in range(len(Laplacian))])
group_lap = group_lap.reshape(size,25)

mean_val= []
median_val = []
mean_lp_val = []
median_lp_val = []
all_fin = []
all_fin2 = []
import time
for idx,sets in enumerate(zip(group_diff,group_lap)):
    diff_sets,lap_sets = sets
    diff_set = [np.load(diff,allow_pickle=True) for diff in diff_sets]
    lap_set = [np.load(lap,allow_pickle=True) for lap in lap_sets]
    height,width = diff_set[0].shape
    #fig = plt.figure(figsize=(10, 10))
    #ax = fig.add_subplot(111, projection='3d')
    x_cor = np.arange(width)
    y_cor = np.arange(height)
    X, Y = np.meshgrid(x_cor, y_cor)
    fin = []
    Z = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            pixel_values = [diff_set[num][i][j] for num in range(len(diff_set))]
            Z[i][j] =  np.nanmedian(pixel_values)
    #ax.scatter(X, Y, Z, c="blue", alpha=0.5)

    #ax.imshow(img,cmap='gray', alpha=0.5, extent=(0, width,  0,height),origin='lower',zorder=-1)

    # z = np.zeros((height, width,25))
    # for i in range(height):
    #     for j in range(width):
    #         pixel_values = [diff_set[num][i][j] for num in range(len(diff_set))]
    #         z[i][j] = pixel_values
    fin.append(Z)
    """
    ax.set_title(f"Median Difference Set {idx+1}")
    ax.invert_xaxis()
    ax.set_xlabel('width')
    ax.set_ylabel('height')
    ax.set_zlabel('median')
    ax.view_init(azim=77, elev=53)
    """
    fin = np.array(fin)[0]
    fin = fin.flatten()
    fin = fin[np.isnan(fin) == False]
    fin = fin[fin > 0.0]
    #fig_lap = plt.figure(figsize=(10, 10))
    #ax2 = fig_lap.add_subplot(111, projection='3d')
    Z_lap = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            pixel_values = [lap_set[num][i][j] for num in range(len(lap_set))]
            Z_lap[i][j] =  np.nanmedian(pixel_values)
    fin2 = []
    # z_lap = np.zeros((height, width,25))
    # for i in range(height):
    #     for j in range(width):
    #         pixel_values = [lap_set[num][i][j] for num in range(len(lap_set))]
    #         z_lap[i][j] = pixel_values
    fin2.append(Z_lap)
    fin2 = np.array(fin2)[0]
    fin2 = fin2.flatten()
    fin2 = fin2[np.isnan(fin2) == False]
    fin2 = fin2[fin2 > 0.0]
    """
    ax2.scatter(X, Y, Z_lap, c="red", alpha=0.5)
    ax2.set_title(f"Median Laplacian Set {idx+1}")
    ax2.invert_xaxis()
    ax2.set_xlabel('width')
    ax2.set_ylabel('height')
    ax2.set_zlabel('median')
    ax2.view_init(azim=77, elev=53)
    """
    #plt.show()
    """
    mean_val.append(np.mean(fin))
    median_val.append(np.median(fin))
    mean_lp_val.append(np.mean(fin2))
    median_lp_val.append(np.median(fin2))"""
    all_fin.append(fin)
    all_fin2.append(fin2)
    print("finishing {idx+1} set")
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(211)
ax.boxplot(all_fin,vert=True,patch_artist=True)
ax.set_title(f"Boxplot of Difference")
ax.set_xlabel('Set')
ax.set_ylabel('Difference')
ax2 = fig.add_subplot(212)
ax2.boxplot(all_fin2,vert=True,patch_artist=True)
ax2.set_title(f"Boxplot of Laplacian")
ax2.set_xlabel('Set')
ax2.set_ylabel('Laplacian')
ax2.set_ylim(0,5e10)
fig.savefig("boxplot.png", dpi=300, bbox_inches='tight')
"""   
fig3 = plt.figure(figsize=(10, 10))
ax3 = fig3.add_subplot(211)
ax3.bar(range(1,11),mean_val)
ax3.set_title(f"Mean Difference")
ax3.set_xlabel('Set')
ax3.set_ylabel('Mean')
ax4 = fig3.add_subplot(212)
ax4.bar(range(1,11),median_val)
ax4.set_title(f"Median Difference")
ax4.set_xlabel('Set')
ax4.set_ylabel('Median')

    
fig4 = plt.figure(figsize=(10, 10))
ax5 = fig4.add_subplot(211)
ax5.bar(range(1,11),mean_lp_val)
ax5.set_title(f"Mean Laplacian")
ax5.set_xlabel('Set')
ax5.set_ylabel('Mean')
ax6 = fig4.add_subplot(212)
ax6.bar(range(1,11),median_lp_val)
ax6.set_title(f"Median Laplacian")
ax6.set_xlabel('Set')
ax6.set_ylabel('Median')
plt.show()
""" 