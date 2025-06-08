from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
def plot3d():
    data = [np.load(f"./oblique_mask_imgs2/T{i}.npy") for i in range(1,19)]
    for i,pic in enumerate(data):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        pic[pic == 0] = np.max(pic[pic!=0])
        x = np.arange(pic.shape[1])
        y = np.arange(pic.shape[0])
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, pic, cmap='Greys')
        ax.set_zlim(0,130)
        ax.invert_xaxis()
        ax.invert_zaxis()
        ax.view_init(azim=100, elev=53)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Distance')
        plt.show()
        #plt.savefig(f"./oblique_mask_imgs2/T{(i+1)}_3d.png")


def plane_from_points(P1,P2,P3):
    v1 = P3 - P1
    v2 = P2 - P1
    cp = np.cross(v1,v2)
    a, b, c = cp
    d = np.dot(cp, P3)
    return a, b, c, d

def cal_real_function():
    distance = np.array([i*10 for i in range(3,12)])
    file = open("./oblique2_5Points_distance.txt","w")
    for dis in distance:
        p = np.array([[13.982935,dis-5.0,-5.0],  #right bottom
                      [-13.982935,dis+5.0,-5.0], #left bottom
                      [13.982935,dis-5.0,16.0],   #right top
                      [-13.982935,dis+5.0,16.0],  #left top
                      [0.0,dis,5.5]])             #center
        # since right side will not be the sharp corner, we need to shift the point
        p_bar = np.array([[10.97827+3.367004,dis-10.0+3.696388,-5.0],
                          [-10.97827,dis+10.0,-5.0],
                          [10.97827+3.367004,dis-10.0+3.696388,16.0],
                          [-10.97827,dis+10.0,16.0],
                          [0.0,dis,5.5]])

        #a,b,c,d = plane_from_points(p1,p2,p3)
        #a_bar,b_bar,c_bar,d_bar = plane_from_points(p1_bar,p2_bar,p3_bar)
        for point in p:
            distance = np.sqrt(point[0]**2+point[1]**2+point[2]**2)
            file.write(f"{distance},")
        file.write("\n")
        for point in p_bar:
            distance = np.sqrt(point[0]**2+point[1]**2+point[2]**2)
            file.write(f"{distance},")
        file.write("\n")
        #print(f"distance: {dis}")
        #print(f"plane: {a}x + {b}y + {c}z = {d}")
        #print(f"plane_bar: {a_bar}x + {b_bar}y + {c_bar}z = {d_bar}")
    file.close()
if __name__ == "__main__":
    #plot3d()
    cal_real_function()

